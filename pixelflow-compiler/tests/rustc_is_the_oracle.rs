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

// ───────────────────── B2: folds over constant ranges ─────────────────────
//
// A fold (docs/plans/2026-09-25-the-language-is-kernel.md §1.5) is Rust's
// iterator spelling of a reduction, so rustc is its oracle too: the same
// tokens over the same range, as a host iterator. The bodies are chosen so
// every term and every partial result is exact in `f32`, which makes the
// comparison bit-for-bit whatever order the e-graph combines the terms in
// once it unrolls the fold.

/// The points the folds are sampled at: every term below is a multiple of a
/// quarter, small enough that no sum or product of them rounds.
const FOLD_SAMPLES: [(f32, f32); 4] = [(3.0, 5.0), (-2.5, 0.5), (0.25, -1.0), (1.0, 1.0)];

/// A mask as a number, so a quantifier's value can be read off a lattice.
fn indicator(m: bool) -> f32 {
    if m { 1.0 } else { 0.0 }
}

/// `.map(|i| e).sum()` is Σ over the range, as rustc's `Iterator::sum` is.
#[test]
fn a_sum_folds_as_rustcs_does() {
    let k = kernel!(|| (0..5).map(|i| X * (i as f32) + Y).sum());
    let rust = |x: f32, y: f32| -> f32 { (0..5).map(|i| x * (i as f32) + y).sum() };
    for (x, y) in FOLD_SAMPLES {
        assert_eq!(Lattice::eval_at(&k, x, y), rust(x, y), "at ({x}, {y})");
    }
    assert_eq!(bake(&k), 55.0, "5 + 8 + 11 + 14 + 17");
}

/// `.map(|i| e).product()` is Π over the range; a range need not start at 0.
#[test]
fn a_product_folds_as_rustcs_does() {
    let k = kernel!(|| (1..5).map(|i| X + i as f32).product());
    let rust = |x: f32, _y: f32| -> f32 { (1..5).map(|i| x + i as f32).product() };
    for (x, y) in FOLD_SAMPLES {
        assert_eq!(Lattice::eval_at(&k, x, y), rust(x, y), "at ({x}, {y})");
    }
    assert_eq!(bake(&k), 840.0, "4 · 5 · 6 · 7");
}

/// `.fold(f32::INFINITY, f32::min)` is the minimum over the range.
#[test]
fn a_min_folds_as_rustcs_does() {
    let k = kernel!(|| (0..4)
        .map(|i| (X - i as f32).abs() + Y)
        .fold(f32::INFINITY, f32::min));
    let rust = |x: f32, y: f32| {
        (0..4)
            .map(|i| (x - i as f32).abs() + y)
            .fold(f32::INFINITY, f32::min)
    };
    for (x, y) in FOLD_SAMPLES {
        assert_eq!(Lattice::eval_at(&k, x, y), rust(x, y), "at ({x}, {y})");
    }
}

/// `.fold(f32::NEG_INFINITY, f32::max)` is the maximum over the range.
#[test]
fn a_max_folds_as_rustcs_does() {
    let k = kernel!(|| (0..4)
        .map(|i| Y * (i as f32) - X)
        .fold(f32::NEG_INFINITY, f32::max));
    let rust = |x: f32, y: f32| {
        (0..4)
            .map(|i| y * (i as f32) - x)
            .fold(f32::NEG_INFINITY, f32::max)
    };
    for (x, y) in FOLD_SAMPLES {
        assert_eq!(Lattice::eval_at(&k, x, y), rust(x, y), "at ({x}, {y})");
    }
}

/// `.any(|i| m)` is ∃ over the range, of `bool`s, as rustc's `Iterator::any`
/// is: the samples put `X` below every index, between two, and above all.
#[test]
fn any_folds_as_rustcs_does() {
    let k = kernel!(|| if (0..4).any(|i| X < i as f32) {
        1.0
    } else {
        0.0
    });
    let rust = |x: f32, _y: f32| indicator((0..4).any(|i| x < i as f32));
    for (x, y) in FOLD_SAMPLES.into_iter().chain([(3.0, 0.0), (-1.0, 0.0)]) {
        assert_eq!(Lattice::eval_at(&k, x, y), rust(x, y), "at ({x}, {y})");
    }
}

/// `.all(|i| m)` is ∀ over the range, of `bool`s, as rustc's `Iterator::all`
/// is.
#[test]
fn all_folds_as_rustcs_does() {
    let k = kernel!(|| if (1..4).all(|i| Y * (i as f32) > X) {
        1.0
    } else {
        0.0
    });
    let rust = |x: f32, y: f32| indicator((1..4).all(|i| y * (i as f32) > x));
    for (x, y) in FOLD_SAMPLES.into_iter().chain([(2.0, 1.0), (0.5, 1.0)]) {
        assert_eq!(Lattice::eval_at(&k, x, y), rust(x, y), "at ({x}, {y})");
    }
}

/// A fold's body sees the folds around it: `Σ_i Σ_j (10i + jX)` reads the
/// outer index inside the inner body. Each fold binds its own slot, so the
/// inner fold does not capture the outer index — which would compute
/// `Σ_i Σ_j (10j + jX)`, `3 · Σ_j j(10 + X)`, where rustc says
/// `4 · 30 + 3 · 6X`.
#[test]
fn a_nested_fold_captures_the_outer_index_as_rustcs_does() {
    let k = kernel!(|| (0..3)
        .map(|i| (0..4)
            .map(|j| (i as f32) * 10.0 + (j as f32) * X)
            .sum::<f32>())
        .sum());
    let rust = |x: f32, _y: f32| -> f32 {
        (0..3)
            .map(|i| {
                (0..4)
                    .map(|j| (i as f32) * 10.0 + (j as f32) * x)
                    .sum::<f32>()
            })
            .sum()
    };
    for (x, y) in FOLD_SAMPLES {
        assert_eq!(Lattice::eval_at(&k, x, y), rust(x, y), "at ({x}, {y})");
    }
    // X = 3: 4 · (0 + 10 + 20) + 3 · (0 + 3 + 6 + 9) = 120 + 54.
    assert_eq!(bake(&k), 174.0);
    let captured = 3.0 * (0..4).map(|j| (j as f32) * (10.0 + AT.0)).sum::<f32>();
    assert_ne!(bake(&k), captured, "the inner index captured the outer");
}

// A helper that folds, called inside a fold: the route §1.7's glyph takes,
// a piece's helper integrating over its own range inside the sum over the
// pieces. The helper's body is lowered in a frame of its own, but its fold
// is still nested in the caller's.
kernel! {
    fn tens_and_units(x: f32) -> f32 {
        (0..2).map(|j| x * 10.0 + (j as f32)).sum()
    }
    pub fn a_helpers_fold_in_a_fold() -> f32 {
        (0..3).map(|i| tens_and_units(i as f32) * X).sum()
    }
}

fn rust_tens_and_units(x: f32) -> f32 {
    (0..2).map(|j| x * 10.0 + (j as f32)).sum()
}

/// A helper's fold inlined inside a fold is nested in it, as a fold written
/// in place would be, and captures nothing: `Σ_i Σ_j (10i + j) · X` is
/// `Σ_i (20i + 1) · X = 63X`. Were the helper's fold not counted as nested —
/// its index sharing the caller's placeholder — the helper's rename would
/// reach `i` through the argument, and this would compute
/// `Σ_i Σ_j (10j + j) · X = 33X`.
#[test]
fn a_helpers_fold_inside_a_fold_captures_nothing_as_in_rustc() {
    let k = a_helpers_fold_in_a_fold();
    let rust = |x: f32, _y: f32| -> f32 { (0..3).map(|i| rust_tens_and_units(i as f32) * x).sum() };
    for (x, y) in FOLD_SAMPLES {
        assert_eq!(Lattice::eval_at(&k, x, y), rust(x, y), "at ({x}, {y})");
    }
    // X = 3: 63 · 3.
    assert_eq!(bake(&k), 189.0);
    let captured = 3.0 * (0..2).map(|j| (j as f32) * 11.0 * AT.0).sum::<f32>();
    assert_ne!(
        bake(&k),
        captured,
        "the helper's index captured the caller's"
    );
}

/// A fold's index is scoped to the fold's body, as a closure's parameter is:
/// it shadows a `let` and a parameter of its name inside the body, and past
/// the fold the name means them again. Were the index's scope left open past
/// the body — the p6 leak, through a fold — `a` and `r` after the fold would
/// name the index, and the kernel would not compile.
#[test]
fn a_fold_index_shadows_only_inside_its_body() {
    let k = kernel!(|r: f32| {
        let a = X;
        (0..4).map(|a| a as f32).sum::<f32>() + (0..3).map(|r| r as f32).sum::<f32>() * a + r
    })(10.0);
    let rust = |x: f32, r: f32| {
        let a = x;
        (0..4).map(|a| a as f32).sum::<f32>() + (0..3).map(|r| r as f32).sum::<f32>() * a + r
    };
    assert_eq!(bake(&k), rust(AT.0, 10.0));
    // 6 + 3 · X + r.
    assert_eq!(bake(&k), 25.0);
}

/// A fold over an empty range is its monoid's identity, as rustc's is: 0,
/// 1, +∞, −∞, false and true — the sum's `+0.0` where rustc's `Sum for
/// f32` starts from `-0.0`, a difference `==` cannot see.
#[test]
fn a_fold_over_an_empty_range_is_the_identity_as_in_rustc() {
    let sum = kernel!(|| (3..3).map(|i| X + i as f32).sum());
    let product = kernel!(|| (3..3).map(|i| X + i as f32).product());
    let min = kernel!(|| (3..3).map(|i| X + i as f32).fold(f32::INFINITY, f32::min));
    let max = kernel!(|| (3..3)
        .map(|i| X + i as f32)
        .fold(f32::NEG_INFINITY, f32::max));
    let any = kernel!(|| if (3..3).any(|i| X < i as f32) {
        1.0
    } else {
        0.0
    });
    let all = kernel!(|| if (3..3).all(|i| X < i as f32) {
        1.0
    } else {
        0.0
    });

    let x = AT.0;
    let rust_sum: f32 = (3..3).map(|i| x + i as f32).sum();
    let rust_product: f32 = (3..3).map(|i| x + i as f32).product();
    let rust_min = (3..3).map(|i| x + i as f32).fold(f32::INFINITY, f32::min);
    let rust_max = (3..3)
        .map(|i| x + i as f32)
        .fold(f32::NEG_INFINITY, f32::max);
    assert_eq!(bake(&sum), rust_sum);
    assert_eq!(bake(&product), rust_product);
    assert_eq!(bake(&min), rust_min);
    assert_eq!(bake(&max), rust_max);
    assert_eq!(bake(&any), indicator((3..3).any(|i| x < i as f32)));
    assert_eq!(bake(&all), indicator((3..3).all(|i| x < i as f32)));
    assert_eq!(
        [rust_sum, rust_product, rust_min, rust_max],
        [0.0, 1.0, f32::INFINITY, f32::NEG_INFINITY]
    );
}

// A range from `usize` consts, evaluated at expansion as rustc evaluates the
// same `const`s, and a `usize` const converted by `as f32` in the body: the
// mean of `X·i` over `i ∈ [LO, HI)`.
const RUST_LO: usize = 2;
const RUST_TERMS: usize = 3;
const RUST_HI: usize = RUST_LO + RUST_TERMS;
kernel! {
    const LO: usize = 2;
    const TERMS: usize = 3;
    pub const HI: usize = LO + TERMS;
    pub fn mean_over_consts() -> f32 {
        (LO..HI).map(|i| X * (i as f32)).sum::<f32>() / (TERMS as f32)
    }
}

/// A range's bounds may be `usize` consts, and a `pub` one is the host
/// `const`.
#[test]
fn a_range_from_consts_folds_as_rustcs_does() {
    assert_eq!(HI, RUST_HI);
    let k = mean_over_consts();
    let rust = |x: f32, _y: f32| {
        (RUST_LO..RUST_HI).map(|i| x * (i as f32)).sum::<f32>() / (RUST_TERMS as f32)
    };
    for (x, y) in FOLD_SAMPLES {
        assert_eq!(Lattice::eval_at(&k, x, y), rust(x, y), "at ({x}, {y})");
    }
    assert_eq!(bake(&k), 9.0, "3 · (2 + 3 + 4) / 3");
}

// ─────────────── B3: records, and structural counts ───────────────
//
// A record (docs/plans/2026-09-25-the-language-is-kernel.md §1.3) is a Rust
// struct of `f32` fields, and the macro emits it as one: so the oracle for a
// record's field arithmetic is the same tokens over the host struct itself.
// A structural parameter (§1.4) is a Rust const generic, and the oracle for
// a range over one is the same iterator in a host `fn` generic over it.

kernel! {
    /// An affine map of the plane.
    pub struct Affine { pub a: f32, pub b: f32, pub c: f32 }

    fn apply(m: Affine, x: f32, y: f32) -> f32 { m.a * x + m.b * y + m.c }

    /// A record passed on to a helper through an alias, and a field of the
    /// alias read beside it.
    pub fn affine_less(m: Affine, d: f32) -> f32 {
        let n = m;
        apply(n, X, Y) - n.c * d
    }
}

fn rust_apply(m: Affine, x: f32, y: f32) -> f32 {
    m.a * x + m.b * y + m.c
}

/// A record's fields mean what the host struct's fields mean, through an
/// alias and a helper: every value below is a quarter, so each product and
/// sum is exact and the comparison is bit for bit.
#[test]
fn a_records_field_arithmetic_is_rustcs() {
    let maps = [
        Affine {
            a: 1.0,
            b: -2.0,
            c: 0.5,
        },
        Affine {
            a: 0.25,
            b: 3.0,
            c: -1.75,
        },
    ];
    for m in maps {
        for d in [0.0, 2.0, -0.5] {
            let k = affine_less(m, d);
            let rust = |x: f32, y: f32| {
                let n = m;
                rust_apply(n, x, y) - n.c * d
            };
            for (x, y) in FOLD_SAMPLES {
                assert_eq!(
                    Lattice::eval_at(&k, x, y),
                    rust(x, y),
                    "{m:?}, d = {d}, at ({x}, {y})"
                );
            }
        }
    }
}

kernel! {
    const FIRST: usize = 1;

    /// A helper reads its entry's structural parameter as an argument.
    fn per(n: f32, total: f32) -> f32 { total / n }

    /// The mean of `X·i` over `i ∈ [FIRST, FIRST + N)`: `N` in a range's
    /// const arithmetic, and as a value.
    pub fn mean_index<const N: usize>() -> f32 {
        per(N as f32, (FIRST..FIRST + N).map(|i| X * (i as f32)).sum::<f32>())
    }
}

fn rust_mean_index<const N: usize>(x: f32) -> f32 {
    const RUST_FIRST: usize = 1;
    (RUST_FIRST..RUST_FIRST + N)
        .map(|i| x * (i as f32))
        .sum::<f32>()
        / (N as f32)
}

/// A range over a structural parameter, and the parameter as a value, mean
/// what the same tokens in a host `fn` generic over it mean, at each
/// instantiation: each is its own program, and each sum is exact.
#[test]
fn a_structural_count_folds_as_rustcs_const_generic_does() {
    for (x, y) in FOLD_SAMPLES {
        assert_eq!(
            Lattice::eval_at(&mean_index::<1>(), x, y),
            rust_mean_index::<1>(x)
        );
        assert_eq!(
            Lattice::eval_at(&mean_index::<3>(), x, y),
            rust_mean_index::<3>(x)
        );
        assert_eq!(
            Lattice::eval_at(&mean_index::<4>(), x, y),
            rust_mean_index::<4>(x)
        );
    }
    // X = 3: (1 + 2 + 3 + 4) · 3 / 4.
    assert_eq!(bake(&mean_index::<4>()), 7.5);
}

// ─────────────── B3: families, and tuple lets ───────────────
//
// A family (docs/plans/2026-09-25-the-language-is-kernel.md §1.6) is written
// as a Rust array of the block's records or of `f32`s, and iterated as Rust
// iterates one, by value: `masses.into_iter().map(|m| e)`. So rustc is its
// oracle too: the same tokens, over the host struct the macro emits, in a
// host `fn` generic over the same count — the items of the block written
// again as Rust — and, for the closure form, a Rust closure over the same
// array. Every value is a quarter and every partial result exact, so the
// comparison is bit for bit whatever order the optimizer combines the
// copies in.

kernel! {
    /// A weighted point.
    pub struct Mass { pub x: f32, pub w: f32 }

    fn pull(m: Mass, x: f32) -> f32 { m.w * (x - m.x) }

    /// Σ over the family, each element through a helper.
    pub fn total_pull<const N: usize>(masses: [Mass; N]) -> f32 {
        masses.into_iter().map(|m| pull(m, X) + Y).sum()
    }

    /// The nearest, weighted: a minimum over the family.
    pub fn nearest<const N: usize>(masses: [Mass; N]) -> f32 {
        masses
            .into_iter()
            .map(|m| (X - m.x).abs() * m.w)
            .fold(f32::INFINITY, f32::min)
    }

    /// Whether any point is right of the sample, as a number.
    pub fn any_right<const N: usize>(masses: [Mass; N]) -> f32 {
        if masses.into_iter().any(|m| X < m.x) { 1.0 } else { 0.0 }
    }

    /// A family iterated in a fold's body: each copy reads the fold's index,
    /// and a term of it that reads no element, which the copies share.
    pub fn swept<const N: usize>(masses: [Mass; N]) -> f32 {
        (0..3)
            .map(|i| {
                let lever = i as f32 * 2.0 + X;
                masses.into_iter().map(|m| m.w * lever - m.x).sum::<f32>()
            })
            .sum()
    }
}

fn rust_pull(m: Mass, x: f32) -> f32 {
    m.w * (x - m.x)
}

fn rust_total_pull<const N: usize>(masses: [Mass; N], x: f32, y: f32) -> f32 {
    masses.into_iter().map(|m| rust_pull(m, x) + y).sum()
}

fn rust_nearest<const N: usize>(masses: [Mass; N], x: f32) -> f32 {
    masses
        .into_iter()
        .map(|m| (x - m.x).abs() * m.w)
        .fold(f32::INFINITY, f32::min)
}

fn rust_any_right<const N: usize>(masses: [Mass; N], x: f32) -> f32 {
    indicator(masses.into_iter().any(|m| x < m.x))
}

fn rust_swept<const N: usize>(masses: [Mass; N], x: f32) -> f32 {
    (0..3)
        .map(|i| {
            let lever = i as f32 * 2.0 + x;
            masses.into_iter().map(|m| m.w * lever - m.x).sum::<f32>()
        })
        .sum()
}

/// Three weighted points, each coordinate and weight a quarter.
const MASSES: [Mass; 3] = [
    Mass { x: 1.0, w: 2.0 },
    Mass { x: -0.5, w: 0.25 },
    Mass { x: 2.5, w: 1.5 },
];

/// A family of records folds as the same tokens over a Rust array of the
/// host struct do: a sum through a helper, a minimum and `any`, at three
/// points, at one, and at none — each monoid's identity, as rustc's, the
/// sum's `+0.0` where rustc's is `-0.0`, as B2's empty range has it.
#[test]
fn a_family_of_records_folds_as_rustcs_array_does() {
    let [first, ..] = MASSES;
    for (x, y) in FOLD_SAMPLES.into_iter().chain([(2.0, 1.0), (-1.0, 0.25)]) {
        let at = |k: &Kernel| Lattice::eval_at(k, x, y);
        assert_eq!(at(&total_pull(MASSES)), rust_total_pull(MASSES, x, y));
        assert_eq!(at(&nearest(MASSES)), rust_nearest(MASSES, x));
        assert_eq!(at(&any_right(MASSES)), rust_any_right(MASSES, x));
        assert_eq!(at(&total_pull([first])), rust_total_pull([first], x, y));
        assert_eq!(at(&nearest([first])), rust_nearest([first], x));
        assert_eq!(at(&total_pull([])), rust_total_pull([], x, y));
        assert_eq!(at(&nearest([])), rust_nearest([], x));
        assert_eq!(at(&any_right([])), rust_any_right([], x));
    }
    // X = 3, Y = 5: 2·2 + 5, 0.25·3.5 + 5, 1.5·0.5 + 5.
    assert_eq!(bake(&total_pull(MASSES)), 9.0 + 5.875 + 5.75);
    // `==` cannot tell the zeros apart; the bits can. An empty family sums
    // to `Monoid::SUM`'s identity, `+0.0`, where rustc's `Sum for f32`
    // starts from `-0.0`.
    assert_eq!(bake(&total_pull([])).to_bits(), 0.0_f32.to_bits());
}

/// A family iterated in a fold's body reads the fold's index, and the term
/// its copies share reads it too, as rustc's nested iterators do.
#[test]
fn a_family_in_a_fold_reads_its_index_as_rustc_does() {
    let [first, ..] = MASSES;
    for (x, y) in FOLD_SAMPLES {
        let at = |k: &Kernel| Lattice::eval_at(k, x, y);
        assert_eq!(at(&swept(MASSES)), rust_swept(MASSES, x), "at ({x}, {y})");
        assert_eq!(at(&swept([first])), rust_swept([first], x), "at ({x}, {y})");
    }
}

/// A family of `f32`s in the closure form folds as a Rust closure over the
/// same array does: each of the six reductions.
#[test]
fn a_family_of_f32s_folds_as_rustcs_array_does() {
    let sum = kernel!(|v: [f32; 4]| v.into_iter().map(|e| (X - e) * Y).sum::<f32>());
    let product = kernel!(|v: [f32; 4]| v.into_iter().map(|e| X + e).product::<f32>());
    let max = kernel!(|v: [f32; 4]| v
        .into_iter()
        .map(|e| e * X)
        .fold(f32::NEG_INFINITY, f32::max));
    let min = kernel!(|v: [f32; 4]| v
        .into_iter()
        .map(|e| (e - X) * Y)
        .fold(f32::INFINITY, f32::min));
    let all = kernel!(|v: [f32; 4]| if v.into_iter().all(|e| e < X) {
        1.0
    } else {
        0.0
    });
    let any = kernel!(|v: [f32; 4]| if v.into_iter().any(|e| X < e) {
        1.0
    } else {
        0.0
    });
    let rust_sum =
        |v: [f32; 4], x: f32, y: f32| -> f32 { v.into_iter().map(|e| (x - e) * y).sum() };
    let rust_product =
        |v: [f32; 4], x: f32, _y: f32| -> f32 { v.into_iter().map(|e| x + e).product() };
    let rust_max = |v: [f32; 4], x: f32, _y: f32| {
        v.into_iter()
            .map(|e| e * x)
            .fold(f32::NEG_INFINITY, f32::max)
    };
    let rust_min = |v: [f32; 4], x: f32, y: f32| {
        v.into_iter()
            .map(|e| (e - x) * y)
            .fold(f32::INFINITY, f32::min)
    };
    let rust_all = |v: [f32; 4], x: f32, _y: f32| indicator(v.into_iter().all(|e| e < x));
    let rust_any = |v: [f32; 4], x: f32, _y: f32| indicator(v.into_iter().any(|e| x < e));
    for v in [[0.5, -1.0, 2.25, 0.0], [4.0, 0.25, -0.75, 1.5]] {
        for (x, y) in FOLD_SAMPLES {
            let at = |k: &Kernel| Lattice::eval_at(k, x, y);
            assert_eq!(at(&sum(v)), rust_sum(v, x, y), "{v:?} at ({x}, {y})");
            assert_eq!(
                at(&product(v)),
                rust_product(v, x, y),
                "{v:?} at ({x}, {y})"
            );
            assert_eq!(at(&max(v)), rust_max(v, x, y), "{v:?} at ({x}, {y})");
            assert_eq!(at(&min(v)), rust_min(v, x, y), "{v:?} at ({x}, {y})");
            assert_eq!(at(&all(v)), rust_all(v, x, y), "{v:?} at ({x}, {y})");
            assert_eq!(at(&any(v)), rust_any(v, x, y), "{v:?} at ({x}, {y})");
        }
    }
}

/// `let (a, b) = (e1, e2);` binds each name to its expression, all at once,
/// as Rust's does: `let (a, b) = (b, a);` swaps, where binding one name at a
/// time would give `b` the new `a`, and this would be `0` everywhere.
#[test]
fn a_tuple_let_binds_as_rustcs_does() {
    let k = kernel!(|| {
        let (a, b) = (X, Y * 2.0);
        let (a, b) = (b, a);
        let (c,) = (a - b,);
        c * a
    });
    let rust = |x: f32, y: f32| {
        let (a, b) = (x, y * 2.0);
        let (a, b) = (b, a);
        let (c,) = (a - b,);
        c * a
    };
    for (x, y) in FOLD_SAMPLES {
        assert_eq!(Lattice::eval_at(&k, x, y), rust(x, y), "at ({x}, {y})");
    }
    // X = 3, Y = 5: (10 − 3) · 10.
    assert_eq!(bake(&k), 70.0);
}
