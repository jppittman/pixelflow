// The kernels that reach what the point-shaped rows cannot: a *surviving*
// `Reduce` under a lattice wide enough to have a remainder, so the column fold
// is strip-mined into a main fold and a remainder fold and the `Reduce` is
// carved into both.
//
// Included, not compiled: `examples/byte_probe.rs` prints their bytes through
// the host's own backend, and `emit::tests::sibling_folds` pins their bytes on
// all three backends at explicit lane counts. One definition, so the two
// cannot drift into measuring different kernels under the same name. Only
// `pixelflow_ir`'s public vocabulary is named, since an example is a crate of
// its own.
//
// Written as `//` comments, not `//!`, because an `include!`d file may not
// carry inner doc comments.

use pixelflow_ir::fold::{Binder, Fold, Monoid};
use pixelflow_ir::{ExprArena, ExprId, OpKind};

/// Rows the lattice every sibling-fold kernel is compiled over has. Small:
/// the row fold is not what these kernels are about, but it must exist as a
/// real loop around the columns.
pub const ROWS: u32 = 3;

/// A width with a remainder at every lane count a backend has (4, 8 and 16
/// lanes leave 1, 5 and 5 samples over), so a kernel compiled at it has both
/// a main column fold and a remainder column fold.
pub const REMAINDER_WIDTH: u32 = 37;

/// Pieces the glyph-like kernel's fold visits.
const PIECES: u32 = 5;

/// Terms of the parked-roots kernel: each reads two values its fold's scope
/// does not compute (a row-invariant product and the call-invariant constant
/// it was built from), so this many terms is twice as many roots.
pub const PARKED_TERMS: u64 = 2048;

/// The binder slot every user fold here takes. The lattice's own folds take
/// the first slots no reachable fold or `Var` names, so these kernels' folds
/// push the lattice's up by one, as a glyph's does.
fn binder() -> Binder {
    Binder::from_slot(0).expect("slot 0 exists")
}

/// `x`, `y` and the user fold's binder, as the three leaves a body reads.
fn leaves(a: &mut ExprArena) -> (ExprId, ExprId, ExprId) {
    let x = a.push_var(0);
    let y = a.push_var(1);
    let i = a.push_var(binder().var());
    (x, y, i)
}

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

/// A glyph's shape, at the size the emitter sees it: one `SUM` fold over
/// pieces whose body varies with the column (through `x`, so with the lane
/// too), with a row-invariant term and call-invariant constants it reads from
/// the scopes outside, clamped the way a coverage is.
///
/// `min(|Σ_i smoothstep(clamp(x/4 + y/16 + 5/16 - i/2))|, 1)`
pub fn glyph_like() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let (x, y, i) = leaves(&mut a);
    let [zero, one, two, three] = [0.0, 1.0, 2.0, 3.0].map(|v| a.push_const(v));

    // Row-invariant: what the row fold computes once and the piece fold reads.
    let ky = a.push_const(0.0625);
    let sy = a.push_binary(OpKind::Mul, y, ky);
    let c0 = a.push_const(0.3125);
    let row_edge = a.push_binary(OpKind::Add, sy, c0);

    // Varies with the column, the lane and the piece.
    let kx = a.push_const(0.25);
    let sx = a.push_binary(OpKind::Mul, x, kx);
    let ki = a.push_const(0.5);
    let si = a.push_binary(OpKind::Mul, i, ki);
    let edge = a.push_binary(OpKind::Add, sx, row_edge);
    let t = a.push_binary(OpKind::Sub, edge, si);
    let floor = a.push_binary(OpKind::Max, t, zero);
    let clamped = a.push_binary(OpKind::Min, floor, one);
    let twice = a.push_binary(OpKind::Mul, clamped, two);
    let rest = a.push_binary(OpKind::Sub, three, twice);
    let square = a.push_binary(OpKind::Mul, clamped, clamped);
    let area = a.push_binary(OpKind::Mul, square, rest);

    let fold = Fold::new(Monoid::SUM, binder(), 0..PIECES);
    let total = a.push_reduce(fold, area);
    let magnitude = a.push_unary(OpKind::Abs, total);
    let root = a.push_binary(OpKind::Min, magnitude, one);
    (a, root)
}

/// Two folds that read nothing of each other, both varying with the column,
/// binding the same slot (so one binder `Var` node serves both) over
/// different ranges and monoids; their results are added.
///
/// `Σ_{i<6} (x/2 + i)·(y/8) + min_{i<4} max(x - 3i/4, -2)`
pub fn two_sibling_folds() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let (x, y, i) = leaves(&mut a);

    let half = a.push_const(0.5);
    let sx = a.push_binary(OpKind::Mul, x, half);
    let shifted = a.push_binary(OpKind::Add, sx, i);
    let eighth = a.push_const(0.125);
    let sy = a.push_binary(OpKind::Mul, y, eighth);
    let scaled = a.push_binary(OpKind::Mul, shifted, sy);
    let first = a.push_reduce(Fold::new(Monoid::SUM, binder(), 0..6), scaled);

    let step = a.push_const(0.75);
    let si = a.push_binary(OpKind::Mul, i, step);
    let behind = a.push_binary(OpKind::Sub, x, si);
    let floor = a.push_const(-2.0);
    let bounded = a.push_binary(OpKind::Max, behind, floor);
    let second = a.push_reduce(Fold::new(Monoid::MIN, binder(), 0..4), bounded);

    let root = a.push_binary(OpKind::Add, first, second);
    (a, root)
}

/// One fold over `terms` terms, each reading two values the fold does not
/// compute: `y·c_r` (the row's) and `c_r` itself (the call's), so the
/// enclosing scopes park `2 · terms` roots for it. Every term differs, so
/// nothing the optimizer factors can make them one.
///
/// `Σ_{i<3} Σ_r min(y·c_r, x + i + c_r)`
pub fn parked_roots(terms: u64) -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let (x, y, i) = leaves(&mut a);
    let along = a.push_binary(OpKind::Add, x, i);
    let cells: Vec<ExprId> = (0..terms)
        .map(|r| {
            let c = a.push_const(0.001 * (r + 1) as f32);
            let row_term = a.push_binary(OpKind::Mul, y, c);
            let varying = a.push_binary(OpKind::Add, along, c);
            a.push_binary(OpKind::Min, row_term, varying)
        })
        .collect();
    let body = sum_tree(&mut a, &cells);
    let root = a.push_reduce(Fold::new(Monoid::SUM, binder(), 0..3), body);
    (a, root)
}

/// An `If` inside a fold's body whose mask varies by lane and whose arms are
/// each several transcendentals deep, so each is worth a branch: the guard
/// the emitter puts *in the fold's own scope*, where the point-shaped
/// `if_guard` row's is in the body's.
///
/// With `t = x/5 + 3i/10`:
/// `Σ_{i<4} if t < 3/2 then exp(sin(1.7·t)/10) else sqrt(t² + 1/4)`
pub fn guarded_if_in_fold() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let (x, _, i) = leaves(&mut a);
    let fifth = a.push_const(0.2);
    let sx = a.push_binary(OpKind::Mul, x, fifth);
    let step = a.push_const(0.3);
    let si = a.push_binary(OpKind::Mul, i, step);
    let t = a.push_binary(OpKind::Add, sx, si);

    let limit = a.push_const(1.5);
    let mask = a.push_binary(OpKind::Lt, t, limit);

    let rate = a.push_const(1.7);
    let phase = a.push_binary(OpKind::Mul, t, rate);
    let wave = a.push_unary(OpKind::Sin, phase);
    let tenth = a.push_const(0.1);
    let damped = a.push_binary(OpKind::Mul, wave, tenth);
    let hot = a.push_unary(OpKind::Exp, damped);

    let square = a.push_binary(OpKind::Mul, t, t);
    let quarter = a.push_const(0.25);
    let lifted = a.push_binary(OpKind::Add, square, quarter);
    let cold = a.push_unary(OpKind::Sqrt, lifted);

    let body = a.push_ternary(OpKind::If, mask, hot, cold);
    let root = a.push_reduce(Fold::new(Monoid::SUM, binder(), 0..4), body);
    (a, root)
}
