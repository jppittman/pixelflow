//! A fold written in `kernel!` is the fold `Kernel::over` builds: the same
//! arena, by `pixelflow_ir::key::canonical`, for every monoid
//! (docs/plans/2026-09-25-the-language-is-kernel.md §1.5; B7's equivalence
//! gate, in miniature).
//!
//! What a fold *means* is pinned against rustc in `rustc_is_the_oracle.rs`.
//! This file pins what it *is*: one `Reduce` over a `Fold::Range`, its binder
//! chosen as the builder chooses it — inside-out, the lowest slot no fold in
//! the body binds — so that a program written in the syntax and the same
//! program built with the builder are one program, and share a JIT cache
//! entry. `kernel_raw!` keeps the lowered shape, so each comparison is the
//! front end's word and not the optimizer's.
//!
//! And what it is not: the syntax never unrolls. The e-graph may, when the
//! kernel is baked (`HalveFold`), and extraction chooses whether it does; the
//! last two tests pin that either way the pixels of a baked fold are the
//! pixels of its terms written out.

use pixelflow_compiler::{kernel, kernel_raw};
use pixelflow_core::{Kernel, Lattice};
use pixelflow_ir::key::canonical;
use pixelflow_ir::{ExprArena, ExprId, ExprNode, LatticeShape, Monoid};
use pixelflow_search::runtime::optimize_runtime_arena;

/// Whether two kernels are one program: the same canonical key.
fn assert_same_program(written: &Kernel, built: &Kernel) {
    let (written_arena, written_root) = written.parts();
    let (built_arena, built_root) = built.parts();
    assert_eq!(
        canonical(written_arena, written_root),
        canonical(built_arena, built_root),
        "written: {}\nbuilt:   {}",
        written_arena.display(written_root),
        built_arena.display(built_root),
    );
}

fn x() -> Kernel {
    Kernel::x()
}

fn y() -> Kernel {
    Kernel::y()
}

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

/// `Σ_i Σ_j (iX + j)`: the inner fold takes slot 0 and the outer slot 1,
/// as the builder's do, and the body reads both. It is not `Σ_i Σ_j (jX +
/// j)`, the capture the builder once had (`kernel.rs`'s `BinderScope`).
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
        canonical(arena, root),
        canonical(captured_arena, captured_root)
    );
    assert_eq!(binders(arena, root), [5, 4], "outer slot 1, inner slot 0");
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
/// place is: `Σ_i (Σ_j 10i + j) · X`, the inner fold at slot 0 and the outer
/// at slot 1, as the builder's are. The helper is lowered in a frame of its
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
        canonical(arena, root),
        canonical(captured_arena, captured_root)
    );
    assert_eq!(binders(arena, root), [5, 4], "outer slot 1, inner slot 0");
}

/// The `Var` index of the binder of every `Reduce` reachable from `root`,
/// each fold before the folds in its body.
fn binders(arena: &ExprArena, root: ExprId) -> Vec<u8> {
    let mut seen = vec![false; arena.len()];
    let mut stack = vec![root];
    let mut out = Vec::new();
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        if let ExprNode::Reduce { fold, .. } = arena.node(id) {
            out.push(fold.binder().var());
        }
        stack.extend(arena.children(id));
    }
    out
}

/// The number of `Reduce` nodes reachable from `root`.
fn folds(arena: &ExprArena, root: ExprId) -> usize {
    binders(arena, root).len()
}

/// The lattice the unrolling is read over, wider than a vector of lanes on
/// every tier so the comparison crosses batches.
const FRAME: (usize, usize) = (40, 4);

/// The shape a bake compiles a kernel for: the frame's. `Lattice::bake`
/// compiles at `LatticeShape::new(extent)` (`Manifold::compile`), and the
/// JIT cache optimizes with `optimize_runtime_arena` at that shape, the call
/// the tests below make. `Lattice` exposes neither its extent nor the arena
/// it compiled, so this restates the one and repeats the other.
fn frame_shape() -> LatticeShape {
    LatticeShape::new([FRAME.0, FRAME.1].map(|extent| extent as u32))
}

/// The syntax never unrolls: a fold of four terms is one `Reduce`, whether
/// or not the macro optimizes. The bake's e-graph does: `HalveFold` splits
/// `Σ_{i<4} X·i` into its terms, the arithmetic rules fold their constants,
/// and nothing of the fold is left in what the bake compiles. Every pixel is
/// the pixel of the four terms written out.
#[test]
fn a_baked_fold_is_its_terms_written_out() {
    let raw = kernel_raw!(|| (0..4).map(|i| X * (i as f32)).sum());
    let optimized = kernel!(|| (0..4).map(|i| X * (i as f32)).sum());
    let written_out = kernel!(|| X * 0.0 + X * 1.0 + X * 2.0 + X * 3.0);
    for k in [&raw, &optimized] {
        let (arena, root) = k.parts();
        assert_eq!(folds(arena, root), 1, "{}", arena.display(root));
    }

    // What the bake compiles: the runtime tier's optimization, at the
    // lattice's shape.
    let (arena, root) = raw.parts();
    let baked = optimize_runtime_arena(arena, root, frame_shape()).expect("the runtime tier");
    assert_eq!(folds(&baked.0, baked.1), 0, "{}", baked.0.display(baked.1));

    let lattice = Lattice::frame(FRAME.0, FRAME.1);
    let unrolled = lattice.bake(&written_out);
    for k in [&raw, &optimized] {
        assert_eq!(lattice.bake(k).buffer(), unrolled.buffer());
    }
    // Column 5 of row 2 samples X = 5: 6 · 5.
    assert_eq!(unrolled.buffer()[2 * FRAME.0 + 5], 30.0);
}

/// Whether the e-graph unrolls a fold is extraction's choice (plan §1.5;
/// docs/plans/2026-09-24-one-pipeline.md §1.4), and where it keeps the loop
/// instead, the pixels are still those of the terms written out.
///
/// The test above reads the unrolled path; this one must read the loop, so it
/// asserts that the bake keeps it. If extraction comes to unroll this body
/// too, that assertion fails, and the fix is a body it keeps as a loop — not
/// deleting the assertion, which would leave the loop's pixels unread.
#[test]
fn a_baked_fold_is_its_terms_written_out_whichever_extraction_chooses() {
    let raw = kernel_raw!(|| (0..4).map(|i| X * (i as f32) + Y).sum());
    let optimized = kernel!(|| (0..4).map(|i| X * (i as f32) + Y).sum());
    let written_out = kernel!(|| (X * 0.0 + Y) + (X * 1.0 + Y) + (X * 2.0 + Y) + (X * 3.0 + Y));

    let (arena, root) = raw.parts();
    let baked = optimize_runtime_arena(arena, root, frame_shape()).expect("the runtime tier");
    assert_eq!(
        folds(&baked.0, baked.1),
        1,
        "extraction kept this fold as a loop when this test was written: {}",
        baked.0.display(baked.1)
    );

    let lattice = Lattice::frame(FRAME.0, FRAME.1);
    let unrolled = lattice.bake(&written_out);
    for k in [&raw, &optimized] {
        assert_eq!(lattice.bake(k).buffer(), unrolled.buffer());
    }
    // Column 1 of row 2 samples X = 1, Y = 2: 6 · 1 + 4 · 2.
    assert_eq!(unrolled.buffer()[2 * FRAME.0 + 1], 14.0);
}
