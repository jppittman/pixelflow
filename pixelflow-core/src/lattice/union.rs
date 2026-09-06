//! # A union of index ranges: the domain-side encoding of an extent
//!
//! An extent is a contramap along an inclusion `ι : L' ↪ L`. That map is
//! partial on the frame, and a partial function in a total language has two
//! encodings:
//!
//! - **Totalize it into the range.** `select(mask, value, 0)` — every point of
//!   `L` is visited and the mask decides whether the answer counts. This is
//!   what [`Kernel::select`](pixelflow_ir::Kernel::select) is for, it is the
//!   right encoding when pieces genuinely *overlap*, and this module does not
//!   replace it.
//! - **Restrict the index.** Don't ask outside `L'`. In a pull renderer the
//!   lattice does the asking, so the restriction lives on the index side: a
//!   range of indices, never a coordinate. No mask, no guard, no coherence
//!   question — the loop does not go there.
//!
//! [`Lattice`] already spells one such restriction. What it could not spell is
//! a **sum** of them, which is what this module adds.
//!
//! ## Denotation
//!
//! A [`Union`] over an ambient lattice `L` is a partial map from `L`'s index
//! to a kernel, defined piecewise on pairwise-disjoint rectangles:
//!
//! ```text
//! U  =  ⊔ᵢ (ιᵢ : Lᵢ ↪ L, Kᵢ)        with  Lᵢ ∩ Lⱼ = ∅  for  i ≠ j
//!
//! collapse(U)(p)  =  Kᵢ(coord p)    if  p ∈ Lᵢ
//!                 =  0              if  p ∉ ⋃ᵢ Lᵢ
//! ```
//!
//! Disjointness is what makes the first clause a definition rather than a
//! choice: at most one `i` can answer, so no arbitration — no blend, no
//! painter's order, no `select` — is needed or offered. It is checked when a
//! summand is placed, so an overlapping union is not a thing that exists.
//!
//! The law is the representable functor's, restricted to each summand:
//!
//! ```text
//! index(collapse(U), p)  =  Kᵢ(coord p)     for p ∈ Lᵢ
//! ```
//!
//! and the single-summand case *is* [`Lattice::bake`]: a union whose one range
//! is the whole ambient extent collapses to the same buffer, bit for bit
//! ([`tests::one_summand_over_the_whole_extent_is_a_plain_bake`]).
//!
//! ## What it costs
//!
//! `bake(⊔ᵢ (Lᵢ, K))` with one shared kernel is one loop nest per summand;
//! `bake(⊔ᵢ (Lᵢ, Kᵢ))` with a kernel per summand is a draw list. Both are
//! "one program" in the sense that matters — the loop nest is shaped by the
//! summands and every kernel is compiled, specialized at its own summand's
//! extent, before any sample is taken. What they are not, yet, is one
//! *emitted* loop nest: this stage compiles one program per summand and
//! collapses each over its own range, which is where the win already is
//! (a glyph that visits only its own cell is not asked about the rest of the
//! string). Fusing the summands into a single emitted nest is a later
//! refinement.
//!
//! ## Decomposing a kernel is not free of the last bits
//!
//! A union collapses each summand's kernel as its own program, and e-graph
//! **extraction is a function of the arena and of the lattice shape** — not of
//! the expression alone. The same subexpression compiled alone and compiled
//! inside a larger sum are different (equally valid) schedules; so are the same
//! kernel compiled at 160×24 and at 9×24. Both round differently at the edges,
//! so decomposing one kernel into summands can move samples in the last bits,
//! and it does.
//!
//! That is a property of decomposition, not of this module. The difference is
//! exactly the one you already pay by baking the pieces separately and adding
//! them yourself: measured on a two-glyph text kernel, the union and a plain
//! `bake(A) + bake(B)` disagree with `bake(A + B)` on the *same* 19 of 432
//! samples, by the *same* worst 2.503e-6.
//!
//! What this module owes, and what is pinned bit-exactly, is that **a summand
//! collapses exactly what its own kernel bakes at its own shape**:
//! [`tests::one_summand_over_the_whole_extent_is_a_plain_bake`] here, and
//! `pixelflow-graphics`'s `tests/text_union_identity.rs` for every cell of a
//! decomposed text run. Whether two summands' kernels *ought* to agree in the
//! last bits is the compiler's question, not the union's.
//!
//! CLAUDE.md already licenses this ("within a target, the optimizer may still
//! produce a different answer than the unoptimized code"), and it is sound for
//! the same reason: no value was promised. It would become a miscompile the
//! moment anything promised one.
//!
//! [`Lattice::bake`]: crate::Lattice::bake

use alloc::vec;
use alloc::vec::Vec;

use super::manifold::{BoundManifold, Manifold, PlaneRegion};
use super::{DiscreteManifold, Lattice};
use pixelflow_ir::Kernel;

/// A rectangular sub-lattice: the index range `[x0, x0 + width) × [y0, y0 + rows)`.
///
/// Indices, never coordinates. A summand of a [`Union`] says *which samples of
/// the ambient lattice it answers for*, and where those samples are taken is
/// the ambient lattice's business — a call site that wants to move a kernel
/// wants [`Kernel::at`](pixelflow_ir::Kernel::at), and this type has nowhere
/// to put a coordinate.
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

    /// Whether two ranges share an index. Empty ranges meet nothing.
    #[must_use]
    pub fn meets(&self, other: &Self) -> bool {
        !self.is_empty()
            && !other.is_empty()
            && self.x0 < other.x0 + other.width
            && other.x0 < self.x0 + self.width
            && self.y0 < other.y0 + other.rows
            && other.y0 < self.y0 + self.rows
    }
}

/// **A disjoint union of index ranges, each with the kernel sampled on it** —
/// `⊔ᵢ (Lᵢ, Kᵢ)`. See the [module documentation](self) for the denotation.
///
/// Build one with [`Union::over`] and [`Union::place`]; every `place` checks
/// the new summand against the ones already there, so the disjointness the
/// denotation assumes is a property of the value rather than a promise in a
/// comment.
pub struct Union {
    lattice: Lattice,
    pieces: Vec<(IndexRange, Kernel)>,
}

impl Union {
    /// The empty union over `lattice`: every index collapses to 0.
    ///
    /// # Panics
    ///
    /// Panics if the ambient lattice is not a plane (Z or W extent above 1).
    /// A union decomposes a plane's index; a stack of planes is
    /// [`Lattice::collapse`]'s nest, not a summand.
    #[must_use]
    pub fn over(lattice: Lattice) -> Self {
        assert!(
            lattice.extent[2] == 1 && lattice.extent[3] == 1,
            "Union::over: the ambient lattice must be a plane, not {:?}",
            lattice.extent
        );
        Self {
            lattice,
            pieces: Vec::new(),
        }
    }

    /// The ambient lattice this union decomposes.
    #[must_use]
    pub fn lattice(&self) -> Lattice {
        self.lattice
    }

    /// Summands placed so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// Whether nothing has been placed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Add the summand `(range, kernel)`: `kernel` answers for exactly the
    /// indices in `range` and is never asked about any other.
    ///
    /// # Panics
    ///
    /// Panics if `range` leaves the ambient extent, or if it **meets a summand
    /// already placed**. Overlap is not a blend and not a painter's order: two
    /// kernels claiming one index is a question this type refuses to arbitrate,
    /// and a scene that really wants a pixel to choose wants
    /// [`Kernel::select`](pixelflow_ir::Kernel::select) inside one summand.
    pub fn place(&mut self, range: IndexRange, kernel: &Kernel) {
        if range.is_empty() {
            return;
        }
        let (ex, ey) = (
            self.lattice.extent[0] as usize,
            self.lattice.extent[1] as usize,
        );
        assert!(
            range.x0 + range.width <= ex && range.y0 + range.rows <= ey,
            "Union::place: {range:?} leaves the {ex}x{ey} ambient lattice"
        );
        if let Some((other, _)) = self.pieces.iter().find(|(r, _)| r.meets(&range)) {
            panic!("Union::place: {range:?} overlaps the summand {other:?} already placed");
        }
        self.pieces.push((range, kernel.clone()));
    }

    /// Compile every summand at its own extent, binding nothing.
    ///
    /// Compilation goes through the global cache, exactly as
    /// [`Lattice::bake`] does, so this is the expensive half and
    /// [`CompiledUnion::collapse`] is the per-frame half. Hold the result if
    /// you collapse the same scene more than once.
    ///
    /// # Panics
    ///
    /// Whatever [`Manifold::compile`] and [`Manifold::bind`] panic for — in
    /// particular, a summand whose kernel declares a buffer, since binding
    /// nothing leaves that slot empty.
    #[must_use]
    pub fn compile(&self) -> CompiledUnion {
        let pieces = self
            .pieces
            .iter()
            .map(|(range, kernel)| {
                let extent = [range.width as u32, range.rows as u32, 1, 1];
                (*range, Manifold::compile(kernel, extent).bind(&[]))
            })
            .collect();
        CompiledUnion {
            lattice: self.lattice,
            pieces,
        }
    }

    /// Compile and collapse in one line: [`Lattice::bake`] for a union.
    ///
    /// # Panics
    ///
    /// See [`Union::compile`].
    #[must_use]
    pub fn bake(&self) -> DiscreteManifold {
        self.compile().collapse()
    }
}

/// A [`Union`] with every summand compiled at its own extent — the object a
/// frame collapses through.
pub struct CompiledUnion {
    lattice: Lattice,
    pieces: Vec<(IndexRange, BoundManifold)>,
}

impl CompiledUnion {
    /// Tabulate the union over the ambient lattice: each summand's program
    /// fills its own range, and every index no summand claims stays 0.
    #[must_use]
    pub fn collapse(&self) -> DiscreteManifold {
        let (ex, ey) = (
            self.lattice.extent[0] as usize,
            self.lattice.extent[1] as usize,
        );
        let mut buffer = vec![0.0f32; ex * ey];
        let mut scratch = Vec::new();
        for (range, program) in &self.pieces {
            let origin = [
                self.lattice.origin[0] + range.x0 as f32,
                self.lattice.origin[1] + range.y0 as f32,
                self.lattice.origin[2],
                self.lattice.origin[3],
            ];
            let band = PlaneRegion::from_origin(range.width, range.rows, origin);
            let start = range.y0 * ex + range.x0;
            program.collapse_subrect(band, &mut buffer[start..], ex, &mut scratch);
        }
        DiscreteManifold::new(buffer, ex, ey)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coordinate_kernel() -> Kernel {
        Kernel::x().mul(&Kernel::constant(100.0)).add(&Kernel::y())
    }

    fn frame(width: usize, height: usize) -> Lattice {
        Lattice::frame(width, height, 0.0)
    }

    /// The degenerate union is the plain bake, bit for bit: one summand
    /// covering the whole extent restricts nothing, so
    /// `index(collapse(U)) = index(collapse(f))`.
    #[test]
    fn one_summand_over_the_whole_extent_is_a_plain_bake() {
        let lattice = frame(37, 11);
        let kernel = coordinate_kernel();
        let mut union = Union::over(lattice);
        union.place(IndexRange::new(0, 0, 37, 11), &kernel);
        assert_eq!(
            union.bake().buffer(),
            lattice.bake(&kernel).buffer(),
            "a union of one whole-extent summand is not the plain bake"
        );
    }

    /// A union of column strips tabulates each strip's own kernel and nothing
    /// else: the value at an index names both the coordinate it was sampled at
    /// and which summand answered, so a misplaced strip is visible in the
    /// value.
    #[test]
    fn each_summand_answers_for_its_own_range() {
        let lattice = frame(24, 5);
        let mut union = Union::over(lattice);
        for strip in 0..3usize {
            let tag = Kernel::constant((strip as f32 + 1.0) * 1000.0);
            union.place(
                IndexRange::new(strip * 8, 0, 8, 5),
                &coordinate_kernel().add(&tag),
            );
        }
        let baked = union.bake();
        for row in 0..5usize {
            for col in 0..24usize {
                let strip = col / 8;
                let want = col as f32 * 100.0 + row as f32 + (strip as f32 + 1.0) * 1000.0;
                assert_eq!(baked.buffer()[row * 24 + col], want, "row {row} col {col}");
            }
        }
    }

    /// Indices no summand claims are 0 — the second clause of the denotation,
    /// and the reason a union does not have to cover its ambient lattice.
    #[test]
    fn an_unclaimed_index_collapses_to_zero() {
        let lattice = frame(16, 4);
        let mut union = Union::over(lattice);
        union.place(IndexRange::new(4, 1, 5, 2), &Kernel::constant(7.0));
        let baked = union.bake();
        for row in 0..4usize {
            for col in 0..16usize {
                let claimed = (4..9).contains(&col) && (1..3).contains(&row);
                let want = if claimed { 7.0 } else { 0.0 };
                assert_eq!(baked.buffer()[row * 16 + col], want, "row {row} col {col}");
            }
        }
    }

    /// A summand whose width is not a whole SIMD batch must not spill its
    /// final batch into the neighbour's columns: the neighbour is a *different
    /// program*, so an overhang there is not scratch, it is wrong samples.
    #[test]
    fn a_summand_does_not_write_past_its_own_columns() {
        let lattice = frame(64, 6);
        let mut union = Union::over(lattice);
        // Widths chosen to be coprime with every plausible batch width.
        for (i, (x0, w)) in [(0usize, 13usize), (13, 17), (30, 34)].iter().enumerate() {
            union.place(
                IndexRange::new(*x0, 0, *w, 6),
                &Kernel::constant(i as f32 + 1.0),
            );
        }
        let baked = union.bake();
        for row in 0..6usize {
            for col in 0..64usize {
                let want = if col < 13 {
                    1.0
                } else if col < 30 {
                    2.0
                } else {
                    3.0
                };
                assert_eq!(baked.buffer()[row * 64 + col], want, "row {row} col {col}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "overlaps the summand")]
    fn overlapping_summands_are_refused() {
        let mut union = Union::over(frame(32, 32));
        union.place(IndexRange::new(0, 0, 16, 16), &Kernel::constant(1.0));
        union.place(IndexRange::new(15, 15, 4, 4), &Kernel::constant(2.0));
    }

    /// Touching, not overlapping: `[0,16)` and `[16,32)` share no index.
    #[test]
    fn abutting_summands_are_accepted() {
        let mut union = Union::over(frame(32, 32));
        union.place(IndexRange::new(0, 0, 16, 32), &Kernel::constant(1.0));
        union.place(IndexRange::new(16, 0, 16, 32), &Kernel::constant(2.0));
        assert_eq!(union.len(), 2);
    }

    #[test]
    #[should_panic(expected = "leaves the")]
    fn a_summand_outside_the_ambient_lattice_is_refused() {
        let mut union = Union::over(frame(32, 32));
        union.place(IndexRange::new(30, 0, 4, 4), &Kernel::constant(1.0));
    }

    #[test]
    fn an_empty_range_places_nothing() {
        let mut union = Union::over(frame(8, 8));
        union.place(IndexRange::new(0, 0, 0, 8), &Kernel::constant(1.0));
        assert!(union.is_empty(), "an empty range is not a summand");
        assert!(union.bake().buffer().iter().all(|&v| v == 0.0));
    }

    #[test]
    #[should_panic(expected = "must be a plane")]
    fn a_union_over_a_stack_of_planes_is_refused() {
        let refused = Union::over(Lattice {
            extent: [8, 8, 4, 1],
            origin: [0.0; 4],
        });
        assert_eq!(refused.len(), 0, "unreachable: `over` panics above");
    }
}
