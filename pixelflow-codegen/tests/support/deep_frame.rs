// A frame past what NEON addresses directly, and the two helpers every kernel in
// `sibling_rows.rs` builds with.
//
// Included, not compiled, like `sibling_rows.rs`, which includes this file: the
// byte pins (`emit::tests::sibling_folds`), `examples/byte_probe.rs` and
// `tests/deep_frame.rs` (which runs the kernel) all take `deep_frame` from here,
// so the kernel whose bytes are pinned is the kernel whose values are checked.

use pixelflow_ir::{ExprArena, ExprId, OpKind};

/// `deep_frame`'s term count: past 64 KiB of spill area on NEON, the reach of
/// `ldr q`'s scaled 12-bit displacement (`4095 · 16`), so a vector slot's
/// address is computed into a register first. The spill area alone is past it,
/// so the whole frame is. `tests/deep_frame.rs` asserts the property, not the
/// constant.
pub const DEEP_FRAME_TERMS: usize = 3760;

/// A balanced sum: depth `log2(terms.len())`, so a thousand terms is not a
/// thousand-deep chain for a recursive pass to walk.
fn sum_tree(a: &mut ExprArena, terms: &[ExprId]) -> ExprId {
    match terms {
        [] => panic!("a sum of nothing"),
        [only] => *only,
        _ => {
            let (left, right) = terms.split_at(terms.len() / 2);
            let (left, right) = (sum_tree(a, left), sum_tree(a, right));
            a.push_binary(OpKind::Add, left, right)
        }
    }
}

/// `x` and `y`, the two coordinates, as the leaves a point-shaped kernel reads.
fn coordinates(a: &mut ExprArena) -> (ExprId, ExprId) {
    (a.push_var(0), a.push_var(1))
}

/// More values live at once than any register file holds, and more than a
/// 64 KiB frame holds slots for: `terms` distinct sums `x·cᵢ + y·dⱼ`, each read
/// twice, once by a balanced sum in order and once by a balanced sum in
/// reverse. Whichever sum is computed first leaves every term waiting for the
/// other, and interleaving them leaves half, so no schedule is narrow.
///
/// The sums are built from `⌈√terms⌉` products of each coordinate, not
/// `terms` constants, so the kernel does not overflow NEON's constant pool.
///
/// `(Σ tₖ) · (Σ t_{n-1-k})`, `tₖ = x·c_{k mod s} + y·d_{k div s}`
pub fn deep_frame(terms: usize) -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let (x, y) = coordinates(&mut a);
    let side = terms.isqrt() + 1;
    let along = |a: &mut ExprArena, coordinate, first: f32, step: f32| -> Vec<ExprId> {
        (0..side)
            .map(|k| {
                let c = a.push_const(first + k as f32 * step);
                a.push_binary(OpKind::Mul, coordinate, c)
            })
            .collect()
    };
    let (xs, ys) = (along(&mut a, x, 0.25, 0.125), along(&mut a, y, 0.5, 0.0625));
    let mut live: Vec<ExprId> = (0..terms)
        .map(|k| a.push_binary(OpKind::Add, xs[k % side], ys[k / side]))
        .collect();
    let forward = sum_tree(&mut a, &live);
    live.reverse();
    let backward = sum_tree(&mut a, &live);
    let root = a.push_binary(OpKind::Mul, forward, backward);
    (a, root)
}
