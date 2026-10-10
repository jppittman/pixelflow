// The kernels that reach what the point-shaped rows cannot: a *surviving*
// `Reduce` under a lattice wide enough to have a remainder, so the column fold
// is strip-mined into a main fold and a remainder fold and the `Reduce` is
// carved into both.
//
// Included, not compiled: `examples/byte_probe.rs` prints their bytes through
// the host's own backend, `emit::tests::sibling_folds` pins their bytes on all
// three backends at explicit lane counts, and `tests/golden_selected.rs`,
// `tests/memory_ratchet.rs` and `tests/selection_stays_linear.rs` measure them
// through `compile`, all over `TABLE`, the one list of rows. One definition,
// so none of them can drift into measuring different kernels under the same
// name. Only `pixelflow_ir`'s public vocabulary is named, since an example is a
// crate of its own.
//
// `deep_frame.rs`, included below, carries the helpers the kernels share.
//
// Written as `//` comments, not `//!`, because an `include!`d file may not
// carry inner doc comments.

include!("deep_frame.rs");

use pixelflow_ir::arena::{BufferDecl, BufferIdentity, UniformDecl, UniformIdentity};
use pixelflow_ir::fold::{Binder, Fold, Monoid};

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

/// `coordinate · k + other`: a different value for every `k`, so nothing the
/// optimizer shares can make two operands of a coverage row one.
fn affine(a: &mut ExprArena, coordinate: ExprId, other: ExprId, k: f32) -> ExprId {
    let k = a.push_const(k);
    let scaled = a.push_binary(OpKind::Mul, coordinate, k);
    a.push_binary(OpKind::Add, scaled, other)
}

/// `to_int(v)`, so an integer op reads what an integer op is for.
fn to_int(a: &mut ExprArena, v: ExprId) -> ExprId {
    a.push_unary(OpKind::TruncToInt, v)
}

/// Every op the backends owe as a one-operand instruction, once each, each
/// on an operand of its own: `Neg`, `Sqrt`, `Rsqrt`, `Abs`, `Recip`, `Floor`,
/// `Ceil`, `Round` and the conversions `TruncToInt` and `IntToFloat`.
///
/// `Σ_op op(x·k_op + y)`, the conversions as the round trip `to_float(to_int(·))`.
pub fn unary_ops() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let (x, y) = coordinates(&mut a);
    let ops = [
        OpKind::Neg,
        OpKind::Sqrt,
        OpKind::Rsqrt,
        OpKind::Abs,
        OpKind::Recip,
        OpKind::Floor,
        OpKind::Ceil,
        OpKind::Round,
    ];
    let mut terms: Vec<ExprId> = ops
        .iter()
        .enumerate()
        .map(|(i, &op)| {
            let operand = affine(&mut a, x, y, 0.5 + i as f32);
            a.push_unary(op, operand)
        })
        .collect();
    let operand = affine(&mut a, x, y, 9.5);
    let int = to_int(&mut a, operand);
    terms.push(a.push_unary(OpKind::IntToFloat, int));
    let root = sum_tree(&mut a, &terms);
    (a, root)
}

/// Every two-operand op, once each: the arithmetic, the six comparisons, the
/// integer add, and `BitAnd` and `BitOr` both as the logic of masks (every
/// comparison is consumed by one, and the combined mask by an `If`) and as
/// the bit operations of integer data.
///
/// With `p = 3x/2 + y` and `q = 5y/2 + x`, the mask is
/// `((p < q & p >= x) | (p <= y | p > q)) & (p == x & p != y)`, selecting
/// `min(p + q, p - q) · max(p, q) / q` over `to_float((to_int(p) + to_int(q)) & 255 | 1)`.
pub fn binary_ops() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let (x, y) = coordinates(&mut a);
    let p = affine(&mut a, x, y, 1.5);
    let q = affine(&mut a, y, x, 2.5);

    let lt = a.push_binary(OpKind::Lt, p, q);
    let ge = a.push_binary(OpKind::Ge, p, x);
    let le = a.push_binary(OpKind::Le, p, y);
    let gt = a.push_binary(OpKind::Gt, p, q);
    let eq = a.push_binary(OpKind::Eq, p, x);
    let ne = a.push_binary(OpKind::Ne, p, y);
    let inside = a.push_binary(OpKind::BitAnd, lt, ge);
    let outside = a.push_binary(OpKind::BitOr, le, gt);
    let exact = a.push_binary(OpKind::BitAnd, eq, ne);
    let either = a.push_binary(OpKind::BitOr, inside, outside);
    let mask = a.push_binary(OpKind::BitAnd, either, exact);

    let added = a.push_binary(OpKind::Add, p, q);
    let subtracted = a.push_binary(OpKind::Sub, p, q);
    let smaller = a.push_binary(OpKind::Min, added, subtracted);
    let larger = a.push_binary(OpKind::Max, p, q);
    let product = a.push_binary(OpKind::Mul, smaller, larger);
    let quotient = a.push_binary(OpKind::Div, product, q);

    let (ip, iq) = (to_int(&mut a, p), to_int(&mut a, q));
    let integer_sum = a.push_binary(OpKind::IAdd, ip, iq);
    let low_bits = a.push_const(f32::from_bits(255));
    let masked = a.push_binary(OpKind::BitAnd, integer_sum, low_bits);
    let one_bit = a.push_const(f32::from_bits(1));
    let set = a.push_binary(OpKind::BitOr, masked, one_bit);
    let bits = a.push_unary(OpKind::IntToFloat, set);

    let root = a.push_ternary(OpKind::If, mask, quotient, bits);
    (a, root)
}

/// The shifts, the fused multiply-add and a blend that is not worth a branch:
/// `Shl` and `Shr` by an immediate, `MulAdd`, and an `If` whose arms are a
/// multiply and an add.
///
/// `to_float(to_int(p) << 3 >> 2) · y + x`, blended with `x·2` and `y + 1`
/// by `x < y`
pub fn shift_muladd_blend() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let (x, y) = coordinates(&mut a);
    let p = affine(&mut a, x, y, 1.5);
    let int = to_int(&mut a, p);
    let [three, two] = [3.0, 2.0].map(|v| a.push_const(v));
    let left = a.push_binary(OpKind::Shl, int, three);
    let right = a.push_binary(OpKind::Shr, left, two);
    let shifted = a.push_unary(OpKind::IntToFloat, right);
    let fused = a.push_ternary(OpKind::MulAdd, shifted, y, x);

    let mask = a.push_binary(OpKind::Lt, x, y);
    let doubled = a.push_binary(OpKind::Mul, x, two);
    let one = a.push_const(1.0);
    let bumped = a.push_binary(OpKind::Add, y, one);
    let blend = a.push_ternary(OpKind::If, mask, doubled, bumped);
    let root = a.push_binary(OpKind::Add, fused, blend);
    (a, root)
}

/// The uniform whose block offset is past anything a 12-bit scaled
/// displacement reaches.
pub const FAR_UNIFORM: u64 = 5000;

/// A read of the uniform in `slot`, declaring every slot up to it: a uniform's
/// place in the block is its declaration order.
fn uniform_at(a: &mut ExprArena, slot: u64) -> ExprId {
    let mut declared = None;
    while a.uniforms().len() as u64 <= slot {
        declared = Some(a.declare_uniform(UniformDecl {
            id: UniformIdentity::mint(),
            default: 0.0,
        }));
    }
    a.push_uniform(declared.expect("each slot is read once, so it is not yet declared"))
}

/// Every way a kernel reads memory it was not computed from: a gather whose
/// index varies by lane, a broadcast of the one element a row names, a
/// uniform at the first element of the block and one far into it.
///
/// `table[x] + table[y] + u₀ + u₅₀₀₀`
pub fn memory() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let (x, y) = coordinates(&mut a);
    let table = a.declare_buffer(BufferDecl {
        id: BufferIdentity::mint(),
        width: 64,
        height: 1,
    });
    let gathered = {
        let base = a.push_buffer(table);
        a.push_binary(OpKind::RawGather, base, x)
    };
    let broadcast = {
        let base = a.push_buffer(table);
        a.push_binary(OpKind::RawGather, base, y)
    };
    let first = uniform_at(&mut a, 0);
    let far = uniform_at(&mut a, FAR_UNIFORM);
    let root = sum_tree(&mut a, &[gathered, broadcast, first, far]);
    (a, root)
}

/// The width a row is compiled at, which for two of the three is a fact about
/// the target's lanes and so cannot be one number.
#[derive(Clone, Copy)]
pub enum Width {
    /// One sample: all remainder, no main fold exists.
    One,
    /// Exactly one batch: all main, no remainder fold exists.
    OneBatch,
    /// [`REMAINDER_WIDTH`]: a main fold and a remainder fold.
    Remainder,
}

impl Width {
    /// The columns of the lattice, for a target of `lanes` lanes.
    pub fn columns(self, lanes: u32) -> u32 {
        match self {
            Self::One => 1,
            Self::OneBatch => lanes,
            Self::Remainder => REMAINDER_WIDTH,
        }
    }
}

/// A kernel and the width it is compiled at.
pub struct Row {
    pub name: &'static str,
    pub build: fn() -> (ExprArena, ExprId),
    pub width: Width,
}

fn parked() -> (ExprArena, ExprId) {
    parked_roots(PARKED_TERMS)
}

fn deep() -> (ExprArena, ExprId) {
    deep_frame(DEEP_FRAME_TERMS)
}

/// The glyph-like fold at the three widths that decide how many sibling column
/// folds exist (a remainder alone; a main alone; both), and each other kernel
/// where both exist. After them, the coverage rows: every op the backends owe,
/// every way a kernel reads memory, and a frame past what NEON addresses
/// directly, all at the width with a remainder. The names are the byte pins',
/// and the traffic pins'.
pub const TABLE: [Row; 11] = [
    Row {
        name: "glyph_like_w1",
        build: glyph_like,
        width: Width::One,
    },
    Row {
        name: "glyph_like_wL",
        build: glyph_like,
        width: Width::OneBatch,
    },
    Row {
        name: "glyph_like_w37",
        build: glyph_like,
        width: Width::Remainder,
    },
    Row {
        name: "two_sibling_folds_w37",
        build: two_sibling_folds,
        width: Width::Remainder,
    },
    Row {
        name: "parked_roots_w37",
        build: parked,
        width: Width::Remainder,
    },
    Row {
        name: "guarded_if_in_fold_w37",
        build: guarded_if_in_fold,
        width: Width::Remainder,
    },
    Row {
        name: "unary_ops_w37",
        build: unary_ops,
        width: Width::Remainder,
    },
    Row {
        name: "binary_ops_w37",
        build: binary_ops,
        width: Width::Remainder,
    },
    Row {
        name: "shift_muladd_blend_w37",
        build: shift_muladd_blend,
        width: Width::Remainder,
    },
    Row {
        name: "memory_w37",
        build: memory,
        width: Width::Remainder,
    },
    Row {
        name: "deep_frame_w37",
        build: deep,
        width: Width::Remainder,
    },
];
