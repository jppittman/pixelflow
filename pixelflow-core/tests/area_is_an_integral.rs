//! `Kernel::area` — the pixel integral, built as two folds over intervals —
//! collapsed through the production pipeline and judged by scalar `f64`
//! Rust.
//!
//! Step 3 of docs/plans/2026-09-23-an-integral-is-a-fold.md §8: an integral
//! is a fold whose domain is continuous, and no rule closes one yet, so
//! every integral here reaches legalization and is replaced by its one-point
//! quadrature — the midpoint, weighted by the length. What this file pins is
//! that the integral *means* what it says through that whole route:
//!
//! - on the centred pixel, the integral of a multi-affine kernel is its
//!   centre sample, exactly — so the one-point rule is exact there, and the
//!   expected values below are exact integers;
//! - on intervals that are neither centred nor of unit length, where the
//!   pixel's identities (midpoint 0, length 1) would hide a dropped weight,
//!   a swapped binder or a wrong point;
//! - under differentiation, which needs quadrature *before* the derivative
//!   is lowered, in both compile entries;
//! - under composition, where a nested fold rebinding a slot must not
//!   capture it.
//!
//! The judge is never a pixelflow evaluator: every expected value is an
//! `f64` closure over `(x, y)` or a literal (CLAUDE.md, "a same-form check
//! cannot see a shared-definition bug").
//!
//! The lattice is 13×7: 13 is prime, so no SIMD width divides a row and
//! every row ends in a partial batch.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_core::{Kernel, Lattice};
use pixelflow_ir::{ExprArena, ExprId, ExprNode, Fold, LatticeShape, OpKind};
use pixelflow_search::runtime::optimize_runtime_arena;

/// Columns of the test lattice.
const WIDTH: usize = 13;
/// Rows of the test lattice.
const HEIGHT: usize = 7;

/// `k = X·Y + 3X − 2Y + 5`: multi-affine, so its integral over any
/// axis-aligned box is its value at the box's centre, and every value it
/// takes on this lattice is an integer an `f32` holds exactly.
fn k() -> Kernel {
    let (x, y) = (Kernel::x(), Kernel::y());
    x.mul(&y)
        .add(&x.mul(&Kernel::constant(3.0)))
        .sub(&y.mul(&Kernel::constant(2.0)))
        .add(&Kernel::constant(5.0))
}

/// [`k`], in `f64`.
fn k_f64(x: f64, y: f64) -> f64 {
    x * y + 3.0 * x - 2.0 * y + 5.0
}

/// Bake `kernel` over the test lattice and require every texel to equal
/// `reference(x, y)` exactly, `x` the column and `y` the row.
fn assert_exact(name: &str, kernel: &Kernel, reference: impl Fn(f64, f64) -> f64) {
    let baked = Lattice::frame(WIDTH, HEIGHT).bake(kernel);
    let texels = baked.buffer();
    assert_eq!(texels.len(), WIDTH * HEIGHT, "{name}: one texel per sample");
    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            let got = f64::from(texels[row * WIDTH + col]);
            let want = reference(col as f64, row as f64);
            assert!(
                got == want,
                "{name} at (x={col}, y={row}): kernel {got}, f64 reference {want}"
            );
        }
    }
}

/// **(a) The pixel integral of a multi-affine kernel is its centre value.**
/// Exact: `∫∫_{[-½,½)²} k(x+u, y+v) du dv = k(x, y)` for multi-affine `k`,
/// and so is the one-point rule legalization emits, so this holds with no
/// tolerance — and keeps holding once a rule closes the integral instead.
#[test]
fn the_area_of_a_multi_affine_kernel_is_its_centre_value() {
    assert_exact("area(k)", &k().area(), k_f64);
}

/// **(a′) A derivative of an area.** `∂/∂X ∫∫ k = ∫∫ ∂k/∂X = Y + 3`.
///
/// The derivative cannot be lowered through a fold, so the integral must be
/// replaced by its quadrature first — in both compile entries. `legalize`
/// has always run last; the runtime tier lowers what survives extraction on
/// its own, and before `passes::resolve` it lowered only the `Dwrt`, which
/// refused the fold under it and threw the whole saturation away without a
/// word, the bake still correct because the jit cache fell back to the
/// unoptimized arena. The texels cannot see that; `optimize_runtime_arena`
/// returning `Some` is the check.
#[test]
fn the_derivative_of_an_area_is_the_area_of_the_derivative() {
    let slope = k().area().dx();
    let (arena, root) = slope.parts();
    let shape = LatticeShape::new([WIDTH as u32, HEIGHT as u32]);
    assert!(
        optimize_runtime_arena(arena, root, shape).is_some(),
        "the runtime tier must lower a Dwrt over an integral, not decline it"
    );
    assert_exact("area(k).dx()", &slope, |_x, y| y + 3.0);
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

/// **(b) Bound at construction: `at` is precomposition.** `area(k.at(σ))`
/// integrates the screen pixel under the warped shape, `area(k).at(σ)` the
/// warped pixel; under a scaling `σ` they are different integrals, and so
/// must be different terms. (Step 4's moments make them integrate to
/// different values; a multi-affine `k` cannot, so here both bake to
/// `k(2x, 2y)` — the key inequality is the half that distinguishes them.)
#[test]
fn integrating_a_warp_and_warping_an_integral_are_different_terms() {
    let two = Kernel::constant(2.0);
    let (sx, sy) = (Kernel::x().mul(&two), Kernel::y().mul(&two));
    let screen_pixel = k().at(&sx, &sy).area();
    let warped_pixel = k().area().at(&sx, &sy);

    let key = |kernel: &Kernel| {
        let (arena, root) = kernel.parts();
        pixelflow_ir::key::canonical(arena, root).key
    };
    assert_ne!(key(&screen_pixel), key(&warped_pixel));

    let intervals = |kernel: &Kernel| {
        let (arena, root) = kernel.parts();
        reachable(arena, root)
            .into_iter()
            .filter(|&id| {
                matches!(
                    arena.node(id),
                    ExprNode::Reduce {
                        fold: Fold::Interval(_),
                        ..
                    }
                )
            })
            .count()
    };
    assert_eq!(intervals(&screen_pixel), 2, "two folds, one per axis");
    assert_eq!(intervals(&warped_pixel), 2, "and `at` removed neither");

    assert_exact("area(k.at(2X, 2Y))", &screen_pixel, |x, y| {
        k_f64(2.0 * x, 2.0 * y)
    });
    assert_exact("area(k).at(2X, 2Y)", &warped_pixel, |x, y| {
        k_f64(2.0 * x, 2.0 * y)
    });
}

/// The bits of `∫_{lo}^{hi}` binding `slot`, assembled by hand from
/// `Fold::to_bits`'s documented layout: domain tag 1 at bit 112, the slot at
/// 64, the endpoints' `f32` bit patterns at 32 and 0. `Kernel::area` builds
/// only the centred pixel, so this is how an asymmetric interval reaches a
/// kernel at all.
fn interval(slot: u8, lo: f32, hi: f32) -> Fold {
    let bits = 1u128 << 112
        | u128::from(slot) << 64
        | u128::from(lo.to_bits()) << 32
        | u128::from(hi.to_bits());
    Fold::from_bits(bits).expect("a finite, nonempty interval over a live slot")
}

/// The first binder slot's `Var` index — slot 0 of the reduction space.
const SLOT_0_VAR: u8 = 4;
/// Slot 1's.
const SLOT_1_VAR: u8 = 5;

/// **(c) Off the pixel, where its identities hide nothing.**
///
/// `∫_{u_y ∈ [10,11)} ∫_{u_x ∈ [1,3)} (u_x·X + u_y·Y) du_x du_y
///   = ∫_{10}^{11} (4X + 2·u_y·Y) du_y = 4X + 21Y`.
///
/// The body is affine in each binder, so the one-point rule is exact and the
/// expected values exact integers. Everything the pixel cannot tell apart is
/// apart here: returning the body unintegrated leaves a free binder; swapping
/// which binder gets which interval gives `21X + 4Y`; dropping the length
/// gives `2X + 10.5Y`; a corner point instead of the midpoint gives
/// `2X + 20Y`.
#[test]
fn an_asymmetric_integral_is_its_length_times_its_midpoint_value() {
    let mut a = ExprArena::new();
    let (x, y) = (a.push_var(0), a.push_var(1));
    let (u_x, u_y) = (a.push_var(SLOT_0_VAR), a.push_var(SLOT_1_VAR));
    let ux_x = a.push_binary(OpKind::Mul, u_x, x);
    let uy_y = a.push_binary(OpKind::Mul, u_y, y);
    let body = a.push_binary(OpKind::Add, ux_x, uy_y);
    let inner = a.push_reduce(interval(0, 1.0, 3.0), body);
    let root = a.push_reduce(interval(1, 10.0, 11.0), inner);
    let kernel = Kernel::from_parts(a, root);

    assert_exact("∫∫ (u_x·X + u_y·Y)", &kernel, |x, y| {
        4.0 * x + 21.0 * y
    });
}

/// **(f) A warp holding a fold that rebinds the integral's slot.**
///
/// `Σ_{j<2} (X + j)` binds slot 0 — it is built on its own, so it takes the
/// lowest free slot — and `area`'s inner interval binds slot 0 too. `at`
/// substitutes the sum for `X` inside the integral, so the sum's fold lands
/// *inside* a fold over the same slot. The inner binder shadows: quadrature
/// substituting the integral's midpoint must stop at the sum, whose `j` is
/// its own. Expected `k(2x + 1, y)`; captured, `k(2x, y)`.
#[test]
fn a_warp_whose_fold_rebinds_the_integrals_slot_is_not_captured() {
    let x_plus_j = Kernel::sum_over(2, |j| Kernel::x().add(j));
    let warped = k().area().at(&x_plus_j, &Kernel::y());
    assert_exact("area(k).at(Σ_j (X + j), Y)", &warped, |x, y| {
        k_f64(2.0 * x + 1.0, y)
    });
}

/// **(f) A fold that rebinds the slot of an integral it reads by name.**
///
/// `Kernel::over` chooses its slot without seeing through a reference, so
/// the sum takes slot 0 — the slot `area`'s inner interval binds — and
/// `expand_refs` then splices the integral inside the sum. Peeling or
/// halving the sum substitutes its index into its body; reaching into the
/// integral would give `Σ_i k(x + i, y) + i` instead of `3·k(x, y) + 3`.
#[test]
fn a_fold_over_a_named_integral_does_not_capture_its_binder() {
    let area = k().area().by_ref();
    let summed = Kernel::sum_over(3, |i| area.add(i));
    assert_exact("Σ_i (area(k) + i)", &summed, |x, y| {
        3.0 * k_f64(x, y) + 3.0
    });
}
