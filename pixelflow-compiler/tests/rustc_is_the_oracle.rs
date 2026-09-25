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

// ───────────────────── B1: the language's constructs ─────────────────────
//
// Each construct Phase B1 adds (docs/plans/2026-09-25-the-language-is-kernel.md
// §1.2, §1.3) is Rust syntax, so rustc is its oracle too: the same tokens
// as an `if`, a `fn` and a `const` in host Rust.

/// The points the choices below are sampled at: on each side of every
/// threshold, and on one.
const CHOICE_SAMPLES: [(f32, f32); 5] =
    [(3.0, 5.0), (5.0, 3.0), (3.5, 3.0), (4.0, 3.0), (3.0, 3.0)];

/// `if c { a } else { b }` chooses as rustc's `if` chooses.
#[test]
fn an_if_chooses_as_rustcs_does() {
    let k = kernel!(|| if X < Y { X } else { Y });
    let rust = |x: f32, y: f32| if x < y { x } else { y };
    for (x, y) in CHOICE_SAMPLES {
        assert_eq!(Lattice::eval_at(&k, x, y), rust(x, y), "at ({x}, {y})");
    }
}

/// An `else if` chain is a chain of choices, and a condition may be masks
/// combined with `&`.
#[test]
fn an_else_if_chain_chooses_as_rustcs_does() {
    let k = kernel!(|| {
        let d = X - Y;
        let inside = (d > -1.0) & (d < 1.0);
        if inside {
            d
        } else if d <= -1.0 {
            -1.0
        } else {
            1.0
        }
    });
    let rust = |x: f32, y: f32| {
        let d = x - y;
        let inside = (d > -1.0) & (d < 1.0);
        if inside {
            d
        } else if d <= -1.0 {
            -1.0
        } else {
            1.0
        }
    };
    for (x, y) in CHOICE_SAMPLES {
        assert_eq!(Lattice::eval_at(&k, x, y), rust(x, y), "at ({x}, {y})");
    }
}

// The block below is the host `fn`, `const` and `if` of a glyph's coverage
// clamp (plan §1.7), written once in `kernel!` and once in Rust.
kernel! {
    const SNAP: f32 = 1.0 / 1024.0;
    pub const NEARLY_ONE: f32 = 1.0 - SNAP;

    /// A helper: a function of its argument, inlined at each call.
    fn coverage(f: f32) -> f32 {
        let c = f.abs().min(1.0);
        if c >= NEARLY_ONE { 1.0 } else if c <= SNAP { 0.0 } else { c }
    }

    pub fn clipped(scale: f32) -> f32 {
        coverage((X - Y) * scale)
    }
}

const RUST_SNAP: f32 = 1.0 / 1024.0;
const RUST_NEARLY_ONE: f32 = 1.0 - RUST_SNAP;

fn rust_coverage(f: f32) -> f32 {
    let c = f.abs().min(1.0);
    if c >= RUST_NEARLY_ONE {
        1.0
    } else if c <= RUST_SNAP {
        0.0
    } else {
        c
    }
}

/// A helper and a `const` mean what a host `fn` and `const` of the same
/// tokens mean, and a `pub const` is the host `const`.
#[test]
fn a_helper_and_a_const_mean_what_rusts_do() {
    assert_eq!(NEARLY_ONE, RUST_NEARLY_ONE);
    // The scale puts `(X - Y) * scale` at each arm: past 1, under the snap,
    // and in between.
    for scale in [1.0, 0.25, 0.0001, -0.5, 0.49999] {
        let k = clipped(scale);
        for (x, y) in CHOICE_SAMPLES {
            assert_eq!(
                Lattice::eval_at(&k, x, y),
                rust_coverage((x - y) * scale),
                "at ({x}, {y}) × {scale}"
            );
        }
    }
}

// A `const` is evaluated per operation in `f32`, as rustc evaluates one.
// The first `+ 1.0` ties to even in `f32` and stays at 2²⁴, and the second
// must too; an evaluator carrying the exact sum in `f64` and rounding once at
// the end reaches 2²⁴ + 2. One operation could not tell them apart — a single
// `f64` operation cast once is correctly rounded — so the witness is two.
const RUST_SUM: f32 = 16777216.0 + 1.0 + 1.0;
kernel! {
    pub const SUM: f32 = 16777216.0 + 1.0 + 1.0;
    pub fn shifted() -> f32 { X + SUM }
}

/// A `const` rounds each operation once, in `f32`, as rustc rounds it.
#[test]
fn a_const_is_evaluated_in_f32_as_rustc_evaluates_it() {
    assert_eq!(SUM, RUST_SUM);
    assert_eq!(SUM, 16_777_216.0);
    assert_eq!((16777216.0_f64 + 1.0 + 1.0) as f32, 16_777_218.0);
    assert_eq!(Lattice::eval_at(&shifted(), 0.0, 0.0), RUST_SUM);
}

// A product and a sum are two roundings, never one, as rustc's const
// evaluator never contracts them — and the proc-macro crate is built with
// `-fp-contract=fast` (`.cargo/config.toml`), which would let an FMA in.
// `B * C` is exactly 1 + 2⁻¹¹ + 2⁻²⁴, a tie that rounds to even, 1 + 2⁻¹¹,
// so `+ D` gives 0; an FMA keeps the 2⁻²⁴ and gives that instead.
const RUST_B: f32 = 1.0 + 1.0 / 4096.0;
const RUST_C: f32 = 1.0 + 1.0 / 4096.0;
const RUST_D: f32 = -(1.0 + 1.0 / 2048.0);
const RUST_CONTRACTION: f32 = RUST_B * RUST_C + RUST_D;
kernel! {
    const B: f32 = 1.0 + 1.0 / 4096.0;
    const C: f32 = 1.0 + 1.0 / 4096.0;
    const D: f32 = -(1.0 + 1.0 / 2048.0);
    pub const CONTRACTION: f32 = B * C + D;
    pub fn contracted() -> f32 { X + CONTRACTION }
}

/// A `const` never contracts `b * c + d` into one rounding, as rustc's
/// const evaluator never does.
#[test]
fn a_const_never_contracts_as_rustc_never_does() {
    assert_eq!(CONTRACTION, RUST_CONTRACTION);
    assert_eq!(CONTRACTION, 0.0);
    assert_eq!(
        RUST_B.mul_add(RUST_C, RUST_D),
        1.0 / 16_777_216.0,
        "one rounding: 2^-24"
    );
    assert_eq!(Lattice::eval_at(&contracted(), 0.0, 0.0), RUST_CONTRACTION);
}
