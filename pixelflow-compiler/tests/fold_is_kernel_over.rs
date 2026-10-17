//! A fold written in `kernel!` is the fold `Kernel::over` builds: the same
//! arena, by `pixelflow_ir::key::canonical`, for every monoid
//! (docs/language/2026-09-25-the-language-is-kernel.md §1.5; B7's equivalence
//! gate, in miniature).
//!
//! What a fold *means* is pinned against rustc in `rustc_is_the_oracle.rs`.
//! This file pins what it *is*: one `Reduce` over a `Fold`, its binder chosen
//! as the builder chooses it — inside-out, the lowest slot no fold in the body
//! binds — so that a program written in the syntax and the same program built
//! with the builder are one program, and share a JIT cache entry.
//! `kernel_raw!` keeps the lowered shape, so each comparison is the front
//! end's word and not the optimizer's.
//!
//! And what it is not: the syntax never unrolls. Whether the bake unrolls is
//! the optimizer's choice; the last test pins that either way the pixels of a
//! baked fold are those of its terms written out.

use pixelflow_compiler::{kernel, kernel_raw};
use pixelflow_core::{Kernel, Lattice};
use pixelflow_ir::Monoid;
use pixelflow_ir::key::canonical;

mod common;
use common::{assert_same_program, x, y};

#[test]
fn a_sum_is_sum_over() {
    let written = kernel_raw!(|| (0..4).map(|i| X * (i as f32) + Y).sum());
    let built = Kernel::sum_over(4, |i| x().mul(i).add(&y()));
    assert_same_program(&written, &built);
}

#[test]
fn a_product_is_product_over() {
    let written = kernel_raw!(|| (0..4).map(|i| X + i as f32).product());
    let built = Kernel::product_over(4, |i| x().add(i));
    assert_same_program(&written, &built);
}

#[test]
fn a_min_is_min_over() {
    let written = kernel_raw!(|| (0..4)
        .map(|i| (X - i as f32).abs())
        .fold(f32::INFINITY, f32::min));
    let built = Kernel::min_over(4, |i| x().sub(i).abs());
    assert_same_program(&written, &built);
}

#[test]
fn a_max_is_max_over() {
    let written = kernel_raw!(|| (0..4)
        .map(|i| Y * (i as f32))
        .fold(f32::NEG_INFINITY, f32::max));
    let built = Kernel::max_over(4, |i| y().mul(i));
    assert_same_program(&written, &built);
}

#[test]
fn any_is_any_over() {
    let written = kernel_raw!(|| (0..4).any(|i| X < i as f32));
    let built = Kernel::any_over(4, |i| x().lt(i));
    assert_same_program(&written, &built);
}

#[test]
fn all_is_all_over() {
    let written = kernel_raw!(|| (0..4).all(|i| X < i as f32));
    let built = Kernel::all_over(4, |i| x().lt(i));
    assert_same_program(&written, &built);
}

/// Where a range starts is the fold's, as it is `Kernel::over`'s.
#[test]
fn a_range_that_starts_past_zero_is_over() {
    let written = kernel_raw!(|| (2..7).map(|i| X * (i as f32)).sum());
    let built = Kernel::over(Monoid::SUM, 2..7, |i| x().mul(i));
    assert_same_program(&written, &built);
}

/// The empty range is a fold like any other: its identity is the e-graph's
/// to find (`EmptyFold`), not the syntax's.
#[test]
fn an_empty_range_is_over_an_empty_range() {
    let written = kernel_raw!(|| (3..3).map(|i| X * (i as f32)).sum());
    let built = Kernel::over(Monoid::SUM, 3..3, |i| x().mul(i));
    assert_same_program(&written, &built);
}

/// `Σ_i Σ_j (iX + j)`: the inner fold takes the lowest slot and the outer the
/// next, as the builder's do, and the body reads both. It is not
/// `Σ_i Σ_j (jX + j)`, which a shared slot would compute.
#[test]
fn a_nested_fold_is_nested_over_and_captures_nothing() {
    let written = kernel_raw!(|| (0..3)
        .map(|i| (0..4).map(|j| (i as f32) * X + j as f32).sum::<f32>())
        .sum());
    let built = Kernel::sum_over(3, |i| Kernel::sum_over(4, |j| i.mul(&x()).add(j)));
    assert_same_program(&written, &built);

    let captured = Kernel::sum_over(3, |_| Kernel::sum_over(4, |j| j.mul(&x()).add(j)));
    let (arena, root) = written.parts();
    let (captured_arena, captured_root) = captured.parts();
    assert_ne!(
        canonical(arena, root).key,
        canonical(captured_arena, captured_root).key
    );
}

/// A fold's binder is free of every fold its body reaches, including one
/// bound to a `let` outside it: `s` holds slot 0, so the fold reading `s`
/// takes slot 1 — as `Kernel::sum_over` does, over the same body.
#[test]
fn a_fold_over_a_body_that_reads_another_fold_takes_the_next_slot() {
    let written = kernel_raw!(|| {
        let s = (0..3).map(|j| j as f32).sum();
        (0..4).map(|i| s * (i as f32)).sum()
    });
    let s = Kernel::sum_over(3, Kernel::clone);
    let built = Kernel::sum_over(4, |i| s.mul(i));
    assert_same_program(&written, &built);
}

kernel_raw! {
    /// A fold in a helper, over its argument.
    fn weighted(x: f32) -> f32 {
        (1..4).map(|k| x * (k as f32)).sum()
    }

    pub fn fold_in_a_helper() -> f32 { weighted(X) + weighted(Y) }
}

/// A helper's fold is a fold like any other, and each call is its own.
#[test]
fn a_fold_in_a_helper_is_over() {
    let weighted = |v: Kernel| Kernel::over(Monoid::SUM, 1..4, |k| v.mul(k));
    let built = weighted(x()).add(&weighted(y()));
    let written = fold_in_a_helper();
    assert_same_program(&written, &built);
    // 6X + 6Y.
    assert_eq!(Lattice::eval_at(&written, 3.0, 5.0), 48.0);
}

kernel_raw! {
    /// A fold in a helper, called inside a fold.
    fn tens_and_units(x: f32) -> f32 {
        (0..2).map(|j| x * 10.0 + (j as f32)).sum()
    }

    pub fn fold_in_a_helper_in_a_fold() -> f32 {
        (0..3).map(|i| tens_and_units(i as f32) * X).sum()
    }
}

/// A helper's fold inlined inside a fold is nested in it, as one written in
/// place is: `Σ_i (Σ_j 10i + j) · X`, the inner fold at the lowest slot and the
/// outer at the next, as the builder's are. The helper is lowered in a frame of its
/// own, but the fold depth crosses into it; were it reset there, the
/// helper's index would share the caller's placeholder and its rename would
/// reach `i` through the argument — `Σ_i (Σ_j 10j + j) · X`, the capture.
#[test]
fn a_helpers_fold_inside_a_fold_is_nested_over_and_captures_nothing() {
    let written = fold_in_a_helper_in_a_fold();
    let ten = Kernel::constant(10.0);
    let built = Kernel::sum_over(3, |i| Kernel::sum_over(2, |j| i.mul(&ten).add(j)).mul(&x()));
    assert_same_program(&written, &built);

    let captured = Kernel::sum_over(3, |_| Kernel::sum_over(2, |j| j.mul(&ten).add(j)).mul(&x()));
    let (arena, root) = written.parts();
    let (captured_arena, captured_root) = captured.parts();
    assert_ne!(
        canonical(arena, root).key,
        canonical(captured_arena, captured_root).key
    );
}

/// The lattice the baked folds are read over, wider than a vector of lanes on
/// every tier so the comparison crosses batches.
const FRAME: (usize, usize) = (40, 4);

/// Every pixel of the frame is `Σ_{i<n} (X·i + Y)` for its own `X` and `Y`:
/// `X · n(n−1)/2 + n · Y`. Each term and partial sum is an integer below 2²⁴
/// at this frame, so the comparison is exact in any association order.
fn assert_baked_fold_is_its_terms(kernels: [&Kernel; 2], terms: u32) {
    let n = terms as f32;
    let lattice = Lattice::frame(FRAME.0, FRAME.1);
    for k in kernels {
        let baked = lattice.bake(k);
        for row in 0..FRAME.1 {
            for column in 0..FRAME.0 {
                let want = (column as f32) * (n * (n - 1.0) / 2.0) + n * (row as f32);
                assert_eq!(
                    baked.buffer()[row * FRAME.0 + column],
                    want,
                    "{terms} terms at ({column}, {row})"
                );
            }
        }
    }
}

/// The syntax never unrolls, so a fold's pixels are the pixels of its terms
/// written out whichever way the bake treats it. A short fold and a long one
/// sit on either side of any decision to unroll.
#[test]
fn a_baked_fold_is_its_terms_written_out() {
    let short_raw = kernel_raw!(|| (0..4).map(|i| X * (i as f32) + Y).sum());
    let short = kernel!(|| (0..4).map(|i| X * (i as f32) + Y).sum());
    assert_baked_fold_is_its_terms([&short_raw, &short], 4);

    let long_raw = kernel_raw!(|| (0..256).map(|i| X * (i as f32) + Y).sum());
    let long = kernel!(|| (0..256).map(|i| X * (i as f32) + Y).sum());
    assert_baked_fold_is_its_terms([&long_raw, &long], 256);
}
