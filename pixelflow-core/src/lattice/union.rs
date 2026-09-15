//! # Index ranges: a rectangular restriction of a lattice's index
//!
//! An extent is a contramap along an inclusion `ι : L' ↪ L`. That map is
//! partial on the frame, and a partial function in a total language has two
//! encodings:
//!
//! - **Totalize it into the range.** `select(mask, value, 0)` — every point of
//!   `L` is visited and the mask decides whether the answer counts. This is
//!   what [`Kernel::select`](pixelflow_ir::Kernel::select) is for.
//! - **Restrict the index.** Don't ask outside `L'`. In a pull renderer the
//!   lattice does the asking, so the restriction lives on the index side: a
//!   range of indices, never a coordinate. No mask, no guard, no coherence
//!   question — the loop does not go there.
//!
//! [`IndexRange`] is the second encoding: a rectangular sub-lattice
//! `[x0, x0+width) × [y0, y0+rows)`, used to say *which* samples of an
//! ambient lattice a caller means without ever naming a coordinate.
//! [`Lattice::scanline`] compiles and bakes a kernel over exactly one row's
//! range; the cell-grid renderers meet a caller's requested band against
//! their own grid extent ([`IndexRange::intersect`]) and paint whatever the
//! grid does not cover ([`IndexRange::paint_complement`]).
//!
//! [`Lattice`]: crate::Lattice
//! [`Lattice::scanline`]: crate::Lattice::scanline

use alloc::vec;

use super::DiscreteManifold;
use super::manifold::{Manifold, PlaneRegion};
use pixelflow_ir::Kernel;

/// A rectangular sub-lattice: the index range `[x0, x0 + width) × [y0, y0 + rows)`.
///
/// Indices, never coordinates. A range says *which samples of the ambient
/// lattice a caller means*, and where those samples are taken is the ambient
/// lattice's business — a call site that wants to move a kernel wants
/// [`Kernel::at`](pixelflow_ir::Kernel::at), and this type has nowhere to put
/// a coordinate.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct IndexRange {
    x0: usize,
    y0: usize,
    width: usize,
    rows: usize,
}

impl IndexRange {
    /// The range `[x0, x0 + width) × [y0, y0 + rows)`.
    #[must_use]
    pub fn new(x0: usize, y0: usize, width: usize, rows: usize) -> Self {
        Self {
            x0,
            y0,
            width,
            rows,
        }
    }

    /// First index on X.
    #[must_use]
    pub fn x0(&self) -> usize {
        self.x0
    }

    /// First index on Y.
    #[must_use]
    pub fn y0(&self) -> usize {
        self.y0
    }

    /// Indices along X.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Indices along Y.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Whether the range contains no indices.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.rows == 0
    }

    /// The range of indices in both — empty when they share none.
    ///
    /// The domain-side `∩` a restricted collapse needs: a caller asked for a
    /// band, a program answers for a range, and what it is asked is the meet.
    /// Total rather than optional, because an empty range is a perfectly good
    /// answer and a `None` would only be unwrapped back into one.
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self {
        let x0 = self.x0.max(other.x0);
        let y0 = self.y0.max(other.y0);
        let x1 = (self.x0 + self.width).min(other.x0 + other.width);
        let y1 = (self.y0 + self.rows).min(other.y0 + other.rows);
        Self {
            x0,
            y0,
            width: x1.saturating_sub(x0),
            rows: y1.saturating_sub(y0),
        }
    }

    /// **Paint `value` over every index of this band that `claim` does not
    /// cover** — the constant half of a band split in two, where one kernel
    /// answers for `claim` and a constant answers for the rest (a cell
    /// grid's own extent against a caller's wider band, painted with the
    /// default background).
    ///
    /// `out` is this band's own plane: its first element is the band's first
    /// index and its rows are `stride` elements apart. `claim` must be a
    /// *leading* sub-rectangle — it starts where the band does — which is
    /// what an origin-anchored restriction of a frame band always is.
    ///
    /// Generic in the element because the two consumers differ only there: a
    /// per-channel form writes `f32` coverage and a packed form writes `u32`
    /// pixels. `(out, stride)` is one question asked in two arguments, exactly
    /// as every `collapse_*` in this crate asks it.
    ///
    /// # Panics
    ///
    /// Panics if `claim` is not a leading sub-rectangle of this band, or if
    /// `out` cannot hold the band at `stride`.
    pub fn paint_complement<T: Copy>(&self, claim: Self, out: &mut [T], stride: usize, value: T) {
        assert!(
            stride >= self.width,
            "IndexRange::paint_complement: stride {stride} is narrower than the {} samples a row holds",
            self.width
        );
        let (claim_w, claim_rows) = if claim.is_empty() {
            (0, 0)
        } else {
            assert!(
                claim.x0 == self.x0
                    && claim.y0 == self.y0
                    && claim.width <= self.width
                    && claim.rows <= self.rows,
                "IndexRange::paint_complement: {claim:?} is not a leading sub-rectangle of {self:?}"
            );
            (claim.width, claim.rows)
        };
        for j in 0..self.rows {
            let start = j * stride;
            let row = &mut out[start..start + self.width];
            let from = if j < claim_rows { claim_w } else { 0 };
            row[from..].fill(value);
        }
    }

    /// Bake `kernel` over exactly this range, in isolation: compiled at the
    /// range's own extent (`[width, rows]`) and collapsed starting at its own
    /// index `(x0, y0)`, into a buffer sized exactly `width × height` —
    /// useful standalone when a caller wants one placed piece's samples on
    /// their own rather than folded into something larger.
    /// [`Lattice::scanline`](crate::Lattice::scanline) is this at one row.
    ///
    /// This goes through `BoundManifold::collapse_subrect` rather than
    /// `collapse_rows` because the two differ once a row's final batch
    /// overhangs its declared width, and here that width is this call's own,
    /// not a padded frame's: an overhanging store would run past this
    /// buffer's last column into undefined memory (`width` not a whole SIMD
    /// batch) or the next row (any `width`), neither of which
    /// `collapse_rows`'s "the caller owns the padding" excuse covers.
    ///
    /// # Panics
    ///
    /// Panics if the range is empty, or whatever [`Manifold::compile`] and
    /// [`Manifold::bind`] panic for.
    ///
    /// **Not public API** (`#[doc(hidden)]`):
    /// [`Lattice::scanline`](crate::Lattice::scanline) is the one caller.
    #[doc(hidden)]
    #[must_use]
    pub fn bake(&self, kernel: &Kernel) -> DiscreteManifold {
        assert!(!self.is_empty(), "IndexRange::bake: {self:?} is empty");
        let bound = Manifold::compile(kernel, [self.width as u32, self.rows as u32]).bind(&[]);
        let mut buffer = vec![0.0f32; self.width * self.rows];
        bound.collapse_subrect(
            PlaneRegion::at_index(self.width, self.rows, self.x0, self.y0),
            &mut buffer,
            self.width,
        );
        DiscreteManifold::new(buffer, self.width, self.rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `intersect` is the domain-side meet, and it is total: an empty answer
    /// is a perfectly good one. Exercised directly and away from the origin,
    /// because its only production caller always meets an origin-anchored
    /// grid against an origin-anchored band — which is one case out of many.
    #[test]
    fn intersect_is_the_meet_of_two_ranges() {
        let r = |x0, y0, w, h| IndexRange::new(x0, y0, w, h);
        let cases: [(IndexRange, IndexRange, IndexRange); 9] = [
            // Overlapping, neither containing the other.
            (r(2, 3, 6, 5), r(5, 1, 6, 5), r(5, 3, 3, 3)),
            // `other` left of and above `self`, still meeting.
            (r(5, 5, 4, 4), r(3, 3, 4, 4), r(5, 5, 2, 2)),
            // Containment, both ways.
            (r(0, 0, 10, 10), r(3, 4, 2, 2), r(3, 4, 2, 2)),
            (r(3, 4, 2, 2), r(0, 0, 10, 10), r(3, 4, 2, 2)),
            // Disjoint on X only — the saturating path, and a width of 0.
            (r(0, 0, 4, 9), r(6, 2, 4, 9), r(6, 2, 0, 7)),
            // Disjoint on Y only.
            (r(0, 0, 9, 4), r(2, 6, 9, 4), r(2, 6, 7, 0)),
            // Edge-adjacent: `[0, 4)` and `[4, 8)` share no index.
            (r(0, 0, 4, 4), r(4, 0, 4, 4), r(4, 0, 0, 4)),
            // An empty operand meets nothing.
            (r(0, 0, 8, 8), r(3, 3, 0, 5), r(3, 3, 0, 5)),
            // Both empty.
            (r(1, 1, 0, 0), r(2, 2, 0, 0), r(2, 2, 0, 0)),
        ];
        for (a, b, want) in cases {
            let got = a.intersect(&b);
            assert_eq!(got, want, "{a:?} ∩ {b:?}");
            assert_eq!(
                got.is_empty(),
                want.is_empty(),
                "emptiness of {a:?} ∩ {b:?}"
            );
            // The meet is symmetric in what it contains, even where the
            // empty answer's origin differs.
            assert_eq!(
                b.intersect(&a).is_empty(),
                want.is_empty(),
                "{b:?} ∩ {a:?} disagrees on emptiness"
            );
        }
    }

    /// `paint_complement` fills exactly the complement of the claim and
    /// touches nothing else in the plane — including the stride's spare
    /// columns, which for a caller sharing the destination plane are a
    /// neighbour's.
    #[test]
    fn paint_complement_fills_the_complement_and_only_it() {
        let (w, rows, stride) = (5usize, 3usize, 7usize);
        for (claim_w, claim_rows) in [(3usize, 2usize), (0, 0), (5, 3), (5, 1), (1, 3)] {
            let mut plane = vec![-1.0f32; rows * stride];
            let band = IndexRange::new(0, 2, w, rows);
            let claim = if claim_w == 0 {
                IndexRange::new(9, 9, 0, 0)
            } else {
                IndexRange::new(0, 2, claim_w, claim_rows)
            };
            band.paint_complement(claim, &mut plane, stride, 9.0);
            for j in 0..rows {
                for i in 0..stride {
                    let inside_claim = j < claim_rows && i < claim_w;
                    let want = if i >= w || inside_claim { -1.0 } else { 9.0 };
                    assert_eq!(
                        plane[j * stride + i],
                        want,
                        "plane[{j}][{i}] (claim {claim_w}x{claim_rows}, band {w} wide, stride {stride})"
                    );
                }
            }
        }
    }

    /// A claim that does not start where the band does is a caller mistake,
    /// not a shape to paint around.
    #[test]
    #[should_panic(expected = "leading sub-rectangle")]
    fn paint_complement_refuses_a_claim_that_is_not_leading() {
        let mut plane = vec![0.0f32; 12];
        IndexRange::new(0, 0, 4, 3).paint_complement(
            IndexRange::new(1, 0, 2, 2),
            &mut plane,
            4,
            1.0,
        );
    }
}
