//! D1's gate: the derived rectangle must *contain* the mask's support, and
//! must be worth deriving.
//!
//! The order matters, and it is the plan's
//! ([one conditional, three lowerings](../../docs/plans/2026-09-08-one-conditional-three-lowerings.md) §8).
//!
//! **Soundness is checked directly, not by agreement.** Every test here walks
//! the whole extent, evaluates the mask at each index, and asserts that every
//! nonzero one falls inside the derived rectangle. That is the property a
//! domain split actually depends on, it fails for the right reason, and it
//! does not care whether any hand-written range is correct.
//!
//! Checking only the agreement with `grid_range` would make this a
//! change-detector against an answer whose correctness is assumed — and
//! #1187 is the standing reminder that a hand-written answer in this area can
//! be wrong. So agreement is a *separate* assertion, about usefulness: a
//! rectangle that contains the mask but equals the full extent is sound and
//! worthless, and that is what catches it.
//!
//! The mask is evaluated here in plain Rust rather than through the JIT. That
//! is deliberate and it is not a second definition of the language: these are
//! comparisons of a coordinate against a literal, which is the one corner
//! where `f32` in Rust and the emitted instruction are required to agree
//! exactly (CLAUDE.md pins `Lt`/`Le` as ordered and `Eq`/`Ne` as exact on
//! every target). Nothing here evaluates arithmetic, a transcendental, or any
//! op whose target behaviour diverges.

use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_ir::kind::OpKind;
use pixelflow_ir::{LatticeShape, MaskSupport, mask_support};

/// One axis-aligned literal comparison, as the builder writes it.
#[derive(Clone, Copy)]
struct Cmp {
    axis: u8,
    op: OpKind,
    c: f32,
    /// Written `literal OP axis` instead of `axis OP literal`.
    reversed: bool,
}

fn cmp(axis: u8, op: OpKind, c: f32) -> Cmp {
    Cmp {
        axis,
        op,
        c,
        reversed: false,
    }
}

impl Cmp {
    fn rev(mut self) -> Self {
        self.reversed = true;
        self
    }

    fn push(self, a: &mut ExprArena) -> ExprId {
        let v = a.push_var(self.axis);
        let k = a.push_const(self.c);
        let (l, r) = if self.reversed { (k, v) } else { (v, k) };
        a.push_binary(self.op, l, r)
    }

    /// The truth of this comparison at an integer index, in scalar Rust.
    fn holds_at(self, index: [u32; 2]) -> bool {
        let x = index[self.axis as usize] as f32;
        let (l, r) = if self.reversed {
            (self.c, x)
        } else {
            (x, self.c)
        };
        match self.op {
            OpKind::Lt => l < r,
            OpKind::Le => l <= r,
            OpKind::Gt => l > r,
            OpKind::Ge => l >= r,
            other => panic!("not a comparison: {other:?}"),
        }
    }
}

/// A conjunction of comparisons — the shape every production mask in the
/// symbolic tier's reach has.
fn conjunction(cmps: &[Cmp]) -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let mut root: Option<ExprId> = None;
    for c in cmps {
        let node = c.push(&mut a);
        root = Some(match root {
            None => node,
            Some(prev) => a.push_binary(OpKind::BitAnd, prev, node),
        });
    }
    let root = root.expect("a conjunction needs at least one comparison");
    (a, root)
}

/// **The soundness check.** Walk the whole extent; every index where the mask
/// is nonzero must be inside the derived rectangle.
fn assert_contains_support(cmps: &[Cmp], extent: [u32; 2]) -> MaskSupport {
    let (arena, root) = conjunction(cmps);
    let shape = LatticeShape::new(extent);
    let got = mask_support(&arena, root, shape);

    for y in 0..extent[1] {
        for x in 0..extent[0] {
            let index = [x, y];
            let nonzero = cmps.iter().all(|c| c.holds_at(index));
            assert!(
                !nonzero || got.contains(index),
                "derived support excludes {index:?}, where the mask is nonzero: \
                 start={:?} end={:?}",
                got.start(),
                got.end()
            );
        }
    }
    got
}

#[test]
fn a_half_open_band_on_x_is_exact() {
    let got = assert_contains_support(
        &[cmp(0, OpKind::Ge, 3.0), cmp(0, OpKind::Lt, 9.0)],
        [16, 16],
    );
    assert_eq!(got.start(), [3, 0]);
    assert_eq!(got.end(), [9, 16]);
}

#[test]
fn the_grid_mask_is_exactly_its_grid() {
    // `in_grid` as `pixelflow-core`'s cell grid writes it: X and Y each
    // fenced above by zero and below by the grid's extent in pixels.
    let got = assert_contains_support(
        &[
            cmp(0, OpKind::Ge, 0.0),
            cmp(0, OpKind::Lt, 40.0),
            cmp(1, OpKind::Ge, 0.0),
            cmp(1, OpKind::Lt, 24.0),
        ],
        [64, 64],
    );
    assert_eq!(got.start(), [0, 0]);
    assert_eq!(
        got.end(),
        [40, 24],
        "a grid mask must derive its own grid, or the derivation is sound and worthless"
    );
}

/// The rounding is the one thing that can make this unsound, and the two ends
/// round in opposite directions.
#[test]
fn a_fractional_bound_rounds_outward_on_both_ends() {
    // X < 3.5 admits i = 3, so the end is 4 and not 3.
    let upper = assert_contains_support(&[cmp(0, OpKind::Lt, 3.5)], [16, 16]);
    assert_eq!(upper.end()[0], 4);

    // X >= 3.5 admits i = 4 but not i = 3, so the start is 4 and not 3.
    let lower = assert_contains_support(&[cmp(0, OpKind::Ge, 3.5)], [16, 16]);
    assert_eq!(lower.start()[0], 4);

    // X <= 3.0 admits i = 3, so the end is 4.
    let inclusive = assert_contains_support(&[cmp(0, OpKind::Le, 3.0)], [16, 16]);
    assert_eq!(inclusive.end()[0], 4);

    // X > 3.0 excludes i = 3, so the start is 4.
    let strict = assert_contains_support(&[cmp(0, OpKind::Gt, 3.0)], [16, 16]);
    assert_eq!(strict.start()[0], 4);
}

#[test]
fn a_reversed_comparison_derives_the_same_band() {
    let forward = assert_contains_support(&[cmp(0, OpKind::Lt, 5.0)], [16, 16]);
    // `5.0 > X` is `X < 5.0`, and reading it as `X > 5.0` would exclude every
    // index the mask is actually nonzero on — the failure this pins.
    let reversed = assert_contains_support(&[cmp(0, OpKind::Gt, 5.0).rev()], [16, 16]);
    assert_eq!(forward, reversed);
}

#[test]
fn both_axes_narrow_independently() {
    let got = assert_contains_support(
        &[
            cmp(0, OpKind::Ge, 2.0),
            cmp(0, OpKind::Lt, 6.0),
            cmp(1, OpKind::Ge, 10.0),
            cmp(1, OpKind::Lt, 12.0),
        ],
        [32, 32],
    );
    assert_eq!(got.start(), [2, 10]);
    assert_eq!(got.end(), [6, 12]);
}

#[test]
fn a_bound_outside_the_lattice_clamps_rather_than_overflowing() {
    let got = assert_contains_support(&[cmp(0, OpKind::Lt, 1.0e30)], [16, 16]);
    assert_eq!(got.end()[0], 16);

    let low = assert_contains_support(&[cmp(0, OpKind::Ge, -1.0e30)], [16, 16]);
    assert_eq!(low.start()[0], 0);
}

#[test]
fn a_contradiction_derives_an_empty_rectangle() {
    let got = assert_contains_support(
        &[cmp(0, OpKind::Ge, 9.0), cmp(0, OpKind::Lt, 3.0)],
        [16, 16],
    );
    assert!(
        got.is_empty(),
        "a mask that is nowhere nonzero should derive an empty rectangle, got \
         start={:?} end={:?}",
        got.start(),
        got.end()
    );
}

/// A disjunction's support is not a rectangle, so the derivation returns the
/// hull. Sound, and deliberately not tight.
#[test]
fn a_disjunction_is_the_hull_of_its_arms() {
    let mut a = ExprArena::new();
    let left = cmp(0, OpKind::Lt, 3.0).push(&mut a);
    let right = cmp(0, OpKind::Ge, 12.0).push(&mut a);
    let root = a.push_binary(OpKind::BitOr, left, right);

    let shape = LatticeShape::new([16, 16]);
    let got = mask_support(&a, root, shape);
    assert_eq!(got.start()[0], 0);
    assert_eq!(got.end()[0], 16);

    for x in 0..16u32 {
        // The arms are `X < 3` and `X >= 12`, so the gap `[3, 12)` is where
        // the mask is zero and the hull is deliberately loose.
        let nonzero = !(3..12).contains(&x);
        assert!(!nonzero || got.contains([x, 0]), "hull excludes x={x}");
    }
}

/// Everything the symbolic tier cannot read must widen to the full extent,
/// never to a guess. This is the arm that keeps "sound by construction" true.
#[test]
fn an_unreadable_mask_widens_to_the_whole_lattice() {
    let shape = LatticeShape::new([16, 16]);

    // Arithmetic on the coordinate: the interval tier's business (D4), not
    // this one's.
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let one = a.push_const(1.0);
    let shifted = a.push_binary(OpKind::Add, x, one);
    let five = a.push_const(5.0);
    let root = a.push_binary(OpKind::Lt, shifted, five);
    assert_eq!(
        mask_support(&a, root, shape),
        MaskSupport::everywhere(shape)
    );

    // A comparison between two coordinates names no literal at all.
    let mut b = ExprArena::new();
    let bx = b.push_var(0);
    let by = b.push_var(1);
    let diag = b.push_binary(OpKind::Lt, bx, by);
    assert_eq!(
        mask_support(&b, diag, shape),
        MaskSupport::everywhere(shape)
    );

    // A non-finite literal is refused rather than reasoned about.
    let mut c = ExprArena::new();
    let cx = c.push_var(0);
    let nan = c.push_const(f32::NAN);
    let cmp_nan = c.push_binary(OpKind::Lt, cx, nan);
    assert_eq!(
        mask_support(&c, cmp_nan, shape),
        MaskSupport::everywhere(shape)
    );
}

/// **The pixel-centre trap.** A caller that samples somewhere other than the
/// index leaves arithmetic in the comparison, and the analysis must widen
/// rather than read the literal as if the coordinate were bare.
///
/// This is the one failure mode that *narrows*: with `X = i + ½`, the mask
/// `X >= 3.5` is nonzero from `i = 3`, while reading the literal bare gives
/// `ceil(3.5) = 4` and drops the row. A deleted line of pixels, not a looser
/// box. `cell_grid` really does sample centres, so this shape is production's
/// and not hypothetical.
#[test]
fn a_pixel_centre_shift_widens_instead_of_narrowing() {
    let shape = LatticeShape::new([16, 16]);
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let half = a.push_const(0.5);
    let centre = a.push_binary(OpKind::Add, x, half);
    let c = a.push_const(3.5);
    let root = a.push_binary(OpKind::Ge, centre, c);

    let got = mask_support(&a, root, shape);
    assert_eq!(
        got,
        MaskSupport::everywhere(shape),
        "a shifted coordinate must widen; reading its literal bare would exclude i=3, \
         where the mask is nonzero"
    );

    // The containment property, stated over the shifted sampling directly, so
    // this test fails for the right reason if the arm above is ever relaxed.
    for i in 0..16u32 {
        let nonzero = (i as f32 + 0.5) >= 3.5;
        assert!(
            !nonzero || got.contains([i, 0]),
            "derived support excludes i={i}, where the centre-sampled mask is nonzero"
        );
    }
}

/// A conjunction with one unreadable conjunct keeps every coordinate-only
/// consequence of the others. Discarding the whole predicate because one
/// operand reads a buffer is what would force full-frame blending on exactly
/// the masks that combine geometry with an atlas.
#[test]
fn an_unreadable_conjunct_does_not_poison_its_siblings() {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let forty = a.push_const(40.0);
    let fenced = a.push_binary(OpKind::Lt, x, forty);

    // Stands in for anything the tier cannot read — here, a comparison
    // between two coordinates.
    let y = a.push_var(1);
    let opaque = a.push_binary(OpKind::Lt, x, y);

    let root = a.push_binary(OpKind::BitAnd, fenced, opaque);
    let shape = LatticeShape::new([64, 64]);
    let got = mask_support(&a, root, shape);

    assert_eq!(
        got.end()[0],
        40,
        "the readable conjunct's bound must survive an unreadable sibling"
    );
    assert_eq!(got.end()[1], 64, "the unreadable conjunct bounds nothing");
}
