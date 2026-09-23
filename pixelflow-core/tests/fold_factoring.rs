//! Folds whose bodies carry a factor the binder does not reach, collapsed
//! through the production pipeline and judged by scalar `f64` Rust.
//!
//! This is the gate for `FactorFold` (`pixelflow-search`'s
//! `egraph::fold_rules`), step 2 of docs/plans/2026-09-23-an-integral-is-a-fold.md
//! §8: the runtime tier may now rewrite `⊕_i (c ⊗ f)` into `c ⊗ ⊕_i f`
//! whenever the class variance fact clears `c` of the binder, so a kernel
//! built through the `Kernel` API can come back out of `Lattice::bake`
//! factored. That it still computes the fold the author wrote is this file's
//! business, and a texel check only judges the rule if the texels came from
//! the rule's output — so where today's cost table extracts the factored
//! form, [`assert_extracts_factored`] pins it, and a cost-table change that
//! stops choosing it fails here rather than quietly turning this file into a
//! check of the unfactored fold.
//!
//! The judge is never a pixelflow evaluator. Each kernel is written a second
//! time as an ordinary `f64` closure over `(x, y)`, with `std`'s `sin`/`cos`,
//! and the two are compared texel by texel — a same-form check cannot see a
//! shared-definition bug (CLAUDE.md, "Precision is on the table").
//!
//! The lattice is 37×5: 37 is odd and prime, so no SIMD width divides a row
//! and every row ends in a partial batch.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_core::{Kernel, Lattice, Monoid};
use pixelflow_ir::{ExprArena, ExprId, ExprNode, LatticeShape, OpKind};
use pixelflow_search::runtime::optimize_runtime_arena;

/// Columns of the test lattice — deliberately not a multiple of any lane
/// count.
const WIDTH: usize = 37;
/// Rows of the test lattice.
const HEIGHT: usize = 5;
/// Terms in every fold here.
const TERMS: u32 = 8;

/// f32's rounding, relative to the size of the terms that were combined —
/// not to the result, which a sum of sines can cancel to near zero while
/// every term still carries its own rounding.
const RELATIVE_TOLERANCE: f64 = 1e-4;
/// A floor for texels whose terms are all near zero.
const ABSOLUTE_TOLERANCE: f64 = 1e-5;

/// What one texel should hold, and how large the terms behind it were.
struct Reference {
    value: f64,
    magnitude: f64,
}

/// `Σ_{i < TERMS} term(i)`, in `f64`.
fn sum_of(term: impl Fn(f64) -> f64) -> Reference {
    let terms: Vec<f64> = (0..TERMS).map(|i| term(f64::from(i))).collect();
    Reference {
        value: terms.iter().sum(),
        magnitude: terms.iter().map(|t| t.abs()).sum(),
    }
}

/// `min` or `max`: how two terms combine, and what an empty fold gives.
struct Extremum {
    pick: fn(f64, f64) -> f64,
    identity: f64,
}

const MIN: Extremum = Extremum {
    pick: f64::min,
    identity: f64::INFINITY,
};

const MAX: Extremum = Extremum {
    pick: f64::max,
    identity: f64::NEG_INFINITY,
};

/// `min_{i < TERMS} (offset + f(i))`, in `f64` — the fold as the author
/// wrote it, not as the optimizer may factor it.
fn min_of(offset: f64, f: impl Fn(f64) -> f64) -> Reference {
    extremum(&MIN, offset, f)
}

/// `max_{i < TERMS} (offset + f(i))`, in `f64`.
fn max_of(offset: f64, f: impl Fn(f64) -> f64) -> Reference {
    extremum(&MAX, offset, f)
}

fn extremum(which: &Extremum, offset: f64, f: impl Fn(f64) -> f64) -> Reference {
    let fs: Vec<f64> = (0..TERMS).map(|i| f(f64::from(i))).collect();
    Reference {
        value: fs
            .iter()
            .map(|&fi| offset + fi)
            .fold(which.identity, which.pick),
        magnitude: offset.abs() + fs.iter().fold(0.0_f64, |m, fi| m.max(fi.abs())),
    }
}

/// Bake `kernel` over the test lattice and compare every texel with
/// `reference(x, y)`, where `x` is the column and `y` the row.
fn assert_matches(name: &str, kernel: &Kernel, reference: impl Fn(f64, f64) -> Reference) {
    let baked = Lattice::frame(WIDTH, HEIGHT).bake(kernel);
    let texels = baked.buffer();
    assert_eq!(texels.len(), WIDTH * HEIGHT, "{name}: one texel per sample");
    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            let got = f64::from(texels[row * WIDTH + col]);
            let want = reference(col as f64, row as f64);
            let tolerance = RELATIVE_TOLERANCE * want.magnitude + ABSOLUTE_TOLERANCE;
            assert!(
                (got - want.value).abs() <= tolerance,
                "{name} at (x={col}, y={row}): kernel {got}, f64 reference {}, \
                 |Δ| {} > tolerance {tolerance}",
                want.value,
                (got - want.value).abs()
            );
        }
    }
}

/// Every node reachable from `root`.
fn reachable(arena: &ExprArena, root: ExprId) -> Vec<ExprId> {
    let mut seen = vec![false; arena.len()];
    let mut stack = vec![root];
    let mut found = Vec::new();
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        found.push(id);
        stack.extend(arena.children(id));
    }
    found
}

/// Assert that the term `Lattice::bake` compiles for `kernel` holds
/// `c ⊗ ⊕_i f`: a `distributor` node with a `monoid` fold as an operand.
///
/// The same call `jit_cache` makes, at the same shape, so the same
/// extraction. These kernels are built as `⊕_i (c ⊗ f)`; peeling and halving
/// put the fold's own combiner outside it, never `⊗`, so only `FactorFold`
/// can have produced the shape.
fn assert_extracts_factored(name: &str, kernel: &Kernel, distributor: OpKind, monoid: Monoid) {
    let (arena, root) = kernel.parts();
    let shape = LatticeShape::new([WIDTH as u32, HEIGHT as u32]);
    let optimized = optimize_runtime_arena(arena, root, shape)
        .unwrap_or_else(|| panic!("{name}: the runtime tier declined the kernel"));
    let (arena, root) = (&optimized.0, optimized.1);
    let is_fold = |id: ExprId| matches!(arena.node(id), ExprNode::Reduce { fold, .. } if fold.monoid() == monoid);
    let factored = reachable(arena, root).into_iter().any(|id| {
        matches!(arena.node(id), ExprNode::Binary(op, a, b)
            if op == distributor && (is_fold(a) || is_fold(b)))
    });
    assert!(
        factored,
        "{name}: the extraction no longer holds {distributor:?} outside a {monoid:?} \
         fold, so this test judges the unfactored form and no longer gates \
         FactorFold — find a kernel the extractor does factor"
    );
}

/// `x·0.1 + i` — the phase every sum here reads, so each term depends on
/// both the lattice and the binder.
fn phase(i: &Kernel) -> Kernel {
    Kernel::x().mul(&Kernel::constant(0.1)).add(i)
}

/// `Σ_i (Y + 1) · sin(X·0.1 + i)`: a sum with a scanline-invariant factor on
/// the left.
#[test]
fn a_sum_with_an_invariant_left_factor_matches_f64() {
    let kernel = Kernel::sum_over(TERMS, |i| {
        Kernel::y().add(&Kernel::constant(1.0)).mul(&phase(i).sin())
    });
    assert_matches("Σ (Y+1)·sin(0.1X+i)", &kernel, |x, y| {
        sum_of(|i| (y + 1.0) * (0.1 * x + i).sin())
    });
}

/// `Σ_i sin(Y) · sin(X·0.1 + i)`: a transcendental factor, which the
/// extractor has every reason to keep outside the fold once it may.
#[test]
fn a_sum_with_a_transcendental_factor_matches_f64() {
    let kernel = Kernel::sum_over(TERMS, |i| Kernel::y().sin().mul(&phase(i).sin()));
    assert_extracts_factored("Σ sin(Y)·sin(0.1X+i)", &kernel, OpKind::Mul, Monoid::SUM);
    assert_matches("Σ sin(Y)·sin(0.1X+i)", &kernel, |x, y| {
        sum_of(|i| y.sin() * (0.1 * x + i).sin())
    });
}

/// `Σ_i sin(X·0.1 + i) · √(Y + 2)`: the invariant factor on the right.
#[test]
fn a_sum_with_an_invariant_right_factor_matches_f64() {
    let kernel = Kernel::sum_over(TERMS, |i| {
        phase(i)
            .sin()
            .mul(&Kernel::y().add(&Kernel::constant(2.0)).sqrt())
    });
    assert_extracts_factored("Σ sin(0.1X+i)·√(Y+2)", &kernel, OpKind::Mul, Monoid::SUM);
    assert_matches("Σ sin(0.1X+i)·√(Y+2)", &kernel, |x, y| {
        sum_of(|i| (0.1 * x + i).sin() * (y + 2.0).sqrt())
    });
}

/// `min_i (Y·0.5 + cos(X + i))`: an invariant offset inside a minimum.
#[test]
fn a_min_with_an_invariant_offset_matches_f64() {
    let kernel = Kernel::min_over(TERMS, |i| {
        Kernel::y()
            .mul(&Kernel::constant(0.5))
            .add(&Kernel::x().add(i).cos())
    });
    assert_extracts_factored("min (0.5Y + cos(X+i))", &kernel, OpKind::Add, Monoid::MIN);
    assert_matches("min (0.5Y + cos(X+i))", &kernel, |x, y| {
        min_of(0.5 * y, |i| (x + i).cos())
    });
}

/// `max_i (sin(X·0.3 + i) + cos(Y·0.25))`: an invariant offset on the right
/// of a maximum.
#[test]
fn a_max_with_an_invariant_offset_matches_f64() {
    let kernel = Kernel::max_over(TERMS, |i| {
        Kernel::x()
            .mul(&Kernel::constant(0.3))
            .add(i)
            .sin()
            .add(&Kernel::y().mul(&Kernel::constant(0.25)).cos())
    });
    assert_extracts_factored(
        "max (sin(0.3X+i) + cos(0.25Y))",
        &kernel,
        OpKind::Add,
        Monoid::MAX,
    );
    assert_matches("max (sin(0.3X+i) + cos(0.25Y))", &kernel, |x, y| {
        max_of((0.25 * y).cos(), |i| (0.3 * x + i).sin())
    });
}

/// `Σ_i (Y + sin(X·0.1 + i))`: `Y` is invariant but `Add` does not
/// distribute over `Σ` — the fold is `8·Y + Σ sin`, not `Y + Σ sin`, and a
/// rule that pulled `Y` out as a factor would be off by `7·Y`.
#[test]
fn a_sum_whose_invariant_term_does_not_distribute_matches_f64() {
    let kernel = Kernel::sum_over(TERMS, |i| Kernel::y().add(&phase(i).sin()));
    assert_matches("Σ (Y + sin(0.1X+i))", &kernel, |x, y| {
        sum_of(|i| y + (0.1 * x + i).sin())
    });
}

/// `min_i ((Y − 2) · cos(X + i))`: `Mul` does not distribute over `min` —
/// a negative factor turns the minimum into a maximum, and rows 0 and 1 of
/// the lattice make `Y − 2` negative.
#[test]
fn a_min_whose_invariant_factor_does_not_distribute_matches_f64() {
    let kernel = Kernel::min_over(TERMS, |i| {
        Kernel::y()
            .sub(&Kernel::constant(2.0))
            .mul(&Kernel::x().add(i).cos())
    });
    assert_matches("min ((Y−2)·cos(X+i))", &kernel, |x, y| {
        min_of(0.0, |i| (y - 2.0) * (x + i).cos())
    });
}
