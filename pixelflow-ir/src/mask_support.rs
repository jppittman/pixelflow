//! Where a mask can be nonzero, as a rectangle of lattice indices.
//!
//! Stage **D1** of
//! [one conditional, three lowerings](../../../docs/plans/2026-09-08-one-conditional-three-lowerings.md):
//! derive the range and check it. Nothing here lowers anything — no loop is
//! split, no guard is emitted, and production behaviour is unchanged.
//!
//! # The contract
//!
//! [`mask_support`] returns a rectangle **containing every index at which the
//! mask can be nonzero**. A superset is always admissible and the full extent
//! is the correct answer whenever nothing can be proven, which is what every
//! unrecognised node returns. Soundness therefore has exactly one failure
//! mode — a returned rectangle that *excludes* an index where the mask is
//! nonzero — and that is what the tests assert directly, by collapsing the
//! mask over the whole extent.
//!
//! This is the **symbolic tier**: axis-aligned comparisons against literals,
//! and their conjunctions and disjunctions, read straight off the DAG. Exact,
//! no search. It covers `in_grid` (`pixelflow-core`'s cell grid) and nothing
//! that needs arithmetic on the coordinate.
//!
//! It deliberately does **not** cover a glyph's support, and should not be
//! read as approximating one: a compound glyph applies a full 2×2 affine to
//! each child, which turns the unit-square tests into inequalities mixing X
//! and Y. Those wait for the interval tier (D4).
//!
//! # Why an index and not a coordinate
//!
//! `Lattice::collapse` evaluates the kernel at `X = column`, `Y = row`, as
//! exact integers in `f32`. So where the arena compares a **bare** `Var`
//! against a literal, the map from index to coordinate is the identity, and
//! `X < c` is nonzero exactly on `i < c` over integers, which is
//! `[0, ceil(c))`.
//!
//! The rounding therefore goes **outward on both ends**, and the direction is
//! not symmetric: an upper bound rounds up, a lower bound rounds down. Get one
//! of them backwards and the rectangle excludes a live index, which is the one
//! way this can be wrong.
//!
//! ## The bare `Var` is load-bearing, not an omission
//!
//! A caller may sample somewhere other than the index — the cell grid reads
//! pixel *centres*, `Kernel::at(&(X + ½), &(Y + ½))`, and
//! `cell_grid`'s own border range carries a matching `− ½` for it. `at`
//! substitutes into `Var` leaves, so that arena holds
//! `Lt(Add(Var(0), 0.5), Const(c))`, and [`axis_of`] refuses it: an
//! unrecognised node is the full extent, which is sound.
//!
//! **Do not "improve" [`axis_of`] to see through that `Add`** without moving
//! the shift into the literal at the same time. The failure is silent and it
//! narrows: with `X = i + ½`, the mask `X >= 3.5` is nonzero from `i = 3`,
//! while this module's rule reads `ceil(3.5) = 4` and would drop the row —
//! a deleted line of pixels, not a looser box. Constant folding turning
//! `X + ½ < c` into `X < c − ½` is the *safe* form of the same thing and
//! needs nothing here, because by then the literal really is the threshold on
//! the bare coordinate.
//!
//! # NaN
//!
//! CLAUDE.md's platform table records that `Gt`/`Ge` are unordered on x86
//! (true for a NaN operand) and ordered on aarch64 (false). That divergence
//! cannot reach this analysis: the operand being compared is a coordinate,
//! which is a lattice index and never NaN. A *literal* that is NaN is refused
//! outright rather than reasoned about.

use crate::arena::{COORD_AXES, ExprArena, ExprId, ExprNode};
use crate::kind::OpKind;
use crate::variance::LatticeShape;

/// A rectangle of lattice indices, half-open on each axis.
///
/// Half-open because an extent is a count: `[0, w)` is a full row, and an
/// empty range is `start == end` rather than a special value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskSupport {
    start: [u32; COORD_AXES],
    end: [u32; COORD_AXES],
}

impl MaskSupport {
    /// The whole lattice — the answer when nothing can be proven.
    #[must_use]
    pub fn everywhere(shape: LatticeShape) -> Self {
        Self {
            start: [0; COORD_AXES],
            end: shape.extent(),
        }
    }

    /// First index on each axis.
    #[must_use]
    pub fn start(self) -> [u32; COORD_AXES] {
        self.start
    }

    /// One past the last index on each axis.
    #[must_use]
    pub fn end(self) -> [u32; COORD_AXES] {
        self.end
    }

    /// No index at all: some axis is empty.
    #[must_use]
    pub fn is_empty(self) -> bool {
        (0..COORD_AXES).any(|a| self.start[a] >= self.end[a])
    }

    /// Whether `index` is inside — the predicate the soundness check asks.
    #[must_use]
    pub fn contains(self, index: [u32; COORD_AXES]) -> bool {
        (0..COORD_AXES).all(|a| index[a] >= self.start[a] && index[a] < self.end[a])
    }

    /// Indices in both, which is what a mask conjunction is nonzero on.
    #[must_use]
    fn intersect(self, other: Self) -> Self {
        let mut out = self;
        for a in 0..COORD_AXES {
            out.start[a] = self.start[a].max(other.start[a]);
            out.end[a] = self.end[a].min(other.end[a]).max(out.start[a]);
        }
        out
    }

    /// The **bounding box** of both — a superset of the union, because a
    /// union of rectangles is not one. Sound, and the reason a disjunction of
    /// two far-apart bands is worth no more than the span between them.
    #[must_use]
    fn hull(self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        let mut out = self;
        for a in 0..COORD_AXES {
            out.start[a] = self.start[a].min(other.start[a]);
            out.end[a] = self.end[a].max(other.end[a]);
        }
        out
    }

    /// Narrow one axis to `[lo, hi)`, clamped into the lattice.
    #[must_use]
    fn with_axis(self, axis: usize, lo: u32, hi: u32) -> Self {
        let mut out = self;
        out.start[axis] = self.start[axis].max(lo);
        out.end[axis] = self.end[axis].min(hi).max(out.start[axis]);
        out
    }
}

/// The rectangle outside which `root` is certainly zero, over `shape`.
///
/// Never narrower than the truth: an unrecognised node yields the full
/// extent. See the module docs for what the symbolic tier does and does not
/// reach.
#[must_use]
pub fn mask_support(arena: &ExprArena, root: ExprId, shape: LatticeShape) -> MaskSupport {
    support_of(arena, root, shape, 0)
}

/// A conjunction nests one node per literal, so the recursion is as deep as
/// the predicate is wide. Bounded so a pathological arena cannot blow the
/// stack; exceeding it yields the full extent, which is sound.
const MAX_DEPTH: u32 = 64;

fn support_of(arena: &ExprArena, id: ExprId, shape: LatticeShape, depth: u32) -> MaskSupport {
    let all = MaskSupport::everywhere(shape);
    if depth >= MAX_DEPTH {
        return all;
    }
    match arena.node(id) {
        // A mask conjunction is nonzero only where both operands are, so the
        // rectangle is the intersection. `BitAnd` and not a boolean `And`
        // because a comparison yields an all-ones pattern and the language
        // combines masks bitwise (CLAUDE.md, "Floating point at the edges").
        ExprNode::Binary(OpKind::BitAnd, a, b) => support_of(arena, *a, shape, depth + 1)
            .intersect(support_of(arena, *b, shape, depth + 1)),
        ExprNode::Binary(OpKind::BitOr, a, b) => {
            support_of(arena, *a, shape, depth + 1).hull(support_of(arena, *b, shape, depth + 1))
        }
        ExprNode::Binary(op, a, b) => comparison_support(arena, *op, *a, *b, shape).unwrap_or(all),
        _ => all,
    }
}

/// One axis-aligned comparison, in either operand order.
fn comparison_support(
    arena: &ExprArena,
    op: OpKind,
    lhs: ExprId,
    rhs: ExprId,
    shape: LatticeShape,
) -> Option<MaskSupport> {
    // `axis OP literal`, or the same relation written the other way round.
    // Flipping the operands flips the relation, which is why this is a
    // reversal rather than a second set of arms.
    let (axis, op, literal) = match (axis_of(arena, lhs), constant_of(arena, rhs)) {
        (Some(axis), Some(c)) => (axis, op, c),
        _ => (axis_of(arena, rhs)?, reverse(op)?, constant_of(arena, lhs)?),
    };
    if !literal.is_finite() {
        return None;
    }

    let extent = shape.extent()[axis];
    // Rounding outward, and the two ends round in opposite directions: an
    // upper bound admits every integer strictly below it, a lower bound every
    // integer at or above it. `X < 3.5` is `i <= 3`, so the end is 4;
    // `X >= 3.5` is `i >= 4`, so the start is 4.
    let (lo, hi) = match op {
        OpKind::Lt => (0, ceil_index(literal, extent)),
        OpKind::Le => (
            0,
            floor_index(literal, extent).saturating_add(1).min(extent),
        ),
        OpKind::Gt => (floor_index(literal, extent).saturating_add(1), extent),
        OpKind::Ge => (ceil_index(literal, extent), extent),
        _ => return None,
    };
    Some(MaskSupport::everywhere(shape).with_axis(axis, lo, hi))
}

/// The relation as written with its operands swapped: `c > X` is `X < c`.
fn reverse(op: OpKind) -> Option<OpKind> {
    match op {
        OpKind::Lt => Some(OpKind::Gt),
        OpKind::Le => Some(OpKind::Ge),
        OpKind::Gt => Some(OpKind::Lt),
        OpKind::Ge => Some(OpKind::Le),
        _ => None,
    }
}

/// The coordinate axis this node *is*, if it is one.
///
/// A bare `Var` only. Arithmetic on a coordinate is the interval tier's
/// business (D4); recognising `X + 1 < c` here would mean carrying an affine
/// form, which is a larger D1 than the plan schedules.
fn axis_of(arena: &ExprArena, id: ExprId) -> Option<usize> {
    match arena.node(id) {
        ExprNode::Var(i) if (*i as usize) < COORD_AXES => Some(*i as usize),
        _ => None,
    }
}

fn constant_of(arena: &ExprArena, id: ExprId) -> Option<f32> {
    match arena.node(id) {
        ExprNode::Const(v) => Some(*v),
        _ => None,
    }
}

/// `ceil(v)` as a lattice index, saturating into `[0, extent]`.
fn ceil_index(v: f32, extent: u32) -> u32 {
    if v <= 0.0 {
        return 0;
    }
    let c = libm::ceilf(v);
    if c >= extent as f32 { extent } else { c as u32 }
}

/// `floor(v)` as a lattice index, saturating into `[0, extent]`.
fn floor_index(v: f32, extent: u32) -> u32 {
    if v <= 0.0 {
        return 0;
    }
    let f = libm::floorf(v);
    if f >= extent as f32 { extent } else { f as u32 }
}
