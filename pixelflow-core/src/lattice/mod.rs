//! # Lattice: Representable Functor for Manifold Evaluation
//!
//! A Lattice is a finite domain that collapses a Manifold into a discrete buffer.
//! This is the `tabulate`/`index` pair from representable functors:
//!
//! - **`collapse`** = `tabulate`: `(Rep -> a) -> F a` -- evaluate at every point
//! - **`DiscreteManifold::eval`** = `index`: `F a -> Rep -> a` -- read back by coordinate
//! - **Isomorphism**: `index(collapse(f, domain), i) = f(coord(i))` (up to discretization)
//!
//! Nothing computes until a Lattice demands it. A single-point evaluation is just
//! `PointLattice` -- the degenerate case with all coordinates fixed.
//!
//! The naive implementations here iterate and call `manifold.eval()` per-batch.
//! The JIT-backed fast path will override `collapse` later, reading the DAG,
//! computing variance, and emitting nested loops with hoisting.

/// 1D index-space lattice for feature/tensor operations.
pub mod index;
pub use index::{IndexLattice1D, IndexLattice2D};

use crate::numeric::Numeric;
use crate::{Field, Manifold, PARALLELISM};
use alloc::vec;
use alloc::vec::Vec;

// ============================================================================
// ReduceOp: Binary operators for fold/reduction
// ============================================================================

/// Binary operators for reduction over a lattice domain.
///
/// Each variant carries a monoid: an identity element and an associative binary op.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    /// Additive monoid: identity = 0, op = +
    Add,
    /// Multiplicative monoid: identity = 1, op = *
    Mul,
    /// Min monoid: identity = +inf, op = min
    Min,
    /// Max monoid: identity = -inf, op = max
    Max,
}

impl ReduceOp {
    /// The identity element for this monoid, broadcast to all SIMD lanes.
    #[inline(always)]
    pub(crate) fn identity(self) -> Field {
        match self {
            ReduceOp::Add => Field::from(0.0),
            ReduceOp::Mul => Field::from(1.0),
            ReduceOp::Min => Field::from(f32::INFINITY),
            ReduceOp::Max => Field::from(f32::NEG_INFINITY),
        }
    }

    /// Apply the binary operation: `acc = acc op val`.
    #[inline(always)]
    pub(crate) fn apply(self, acc: Field, val: Field) -> Field {
        match self {
            ReduceOp::Add => acc.raw_add(val),
            ReduceOp::Mul => acc.raw_mul(val),
            ReduceOp::Min => acc.min(val),
            ReduceOp::Max => acc.max(val),
        }
    }
}

// ============================================================================
// LatticeDomain: The `Rep` part of a representable functor
// ============================================================================

/// A finite domain that can generate coordinates.
///
/// This is the `Rep` part of a representable functor. Each index
/// maps to a coordinate tuple that a Manifold can evaluate.
pub trait LatticeDomain {
    /// Number of points in this domain.
    fn len(&self) -> usize;

    /// Whether the domain is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Which coordinate variables are loop dimensions (variance bitset).
    ///
    /// Returns indices of varying coordinates: 0=X, 1=Y, 2=Z, 3=W.
    /// Fixed dimensions have been substituted as constants.
    fn loop_vars(&self) -> &[u8];

    /// Map a linear index to concrete coordinate values.
    ///
    /// Returns `(x, y, z, w)` for this point in the domain.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    fn coord(&self, index: usize) -> (f32, f32, f32, f32);
}

// ============================================================================
// Lattice: The collapse operation
// ============================================================================

/// A Lattice collapses a Manifold over a finite domain.
///
/// `collapse` = `tabulate` in representable functor terms.
/// `collapse_with` = `foldMap` in Foldable terms.
///
/// The result of `collapse` is a `DiscreteManifold` -- a buffer
/// that IS a Manifold (indexable by coordinate). It re-enters the algebra.
pub trait Lattice: LatticeDomain {
    /// Collapse: evaluate the manifold at every point in the domain.
    /// Returns a discrete manifold (buffer lookup).
    ///
    /// This is `tabulate`: `(Rep -> a) -> F a`
    fn collapse<M>(&self, manifold: &M) -> DiscreteManifold
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>;

    /// Fold all points of the lattice into a single value using `op`.
    ///
    /// **Semantic note:** For pixel-space lattices (`FrameLattice`, `ScanlineLattice`),
    /// this returns a per-lane SIMD accumulator where each lane is an independent
    /// parallel result. For index-space lattices (`IndexLattice1D`, `IndexLattice2D`),
    /// this returns a true scalar broadcast to all lanes (horizontal SIMD reduction).
    /// Choose the lattice type that matches your intended semantics.
    fn collapse_with<M>(&self, op: ReduceOp, manifold: &M) -> Field
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>;
}

// ============================================================================
// DiscreteManifold: The result of collapsing a lattice
// ============================================================================

/// The result of collapsing a Lattice. A buffer of values that IS a Manifold.
///
/// `index`: `F a -> Rep -> a` -- read by coordinate.
/// This closes the representable functor isomorphism:
/// `index(collapse(f)) = f` (up to discretization).
#[derive(Clone, Debug)]
pub struct DiscreteManifold {
    /// Raw value buffer, row-major (y * width + x).
    pub(crate) buffer: Vec<f32>,
    /// Width of the grid (X dimension).
    pub(crate) width: usize,
    /// Height of the grid (Y dimension).
    pub(crate) height: usize,
}

impl DiscreteManifold {
    /// Create a DiscreteManifold from a pre-filled buffer.
    ///
    /// # Panics
    ///
    /// Panics if `buffer.len() != width * height`.
    #[must_use]
    pub fn new(buffer: Vec<f32>, width: usize, height: usize) -> Self {
        assert_eq!(
            buffer.len(),
            width * height,
            "DiscreteManifold buffer size {} does not match dimensions {}x{} = {}",
            buffer.len(),
            width,
            height,
            width * height,
        );
        Self {
            buffer,
            width,
            height,
        }
    }

    /// Width of the grid (X extent).
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height of the grid (Y extent).
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Read-only access to the underlying buffer (row-major).
    #[must_use]
    pub fn buffer(&self) -> &[f32] {
        &self.buffer
    }

    /// Mutable access to the underlying buffer (row-major).
    pub fn buffer_mut(&mut self) -> &mut [f32] {
        &mut self.buffer
    }

    /// Consume the DiscreteManifold and return the buffer.
    #[must_use]
    pub fn into_buffer(self) -> Vec<f32> {
        self.buffer
    }
}

// Mark as a ManifoldExpr so ManifoldExt methods (`.at()`, etc.) work on it.
impl crate::ext::ManifoldExpr for DiscreteManifold {}

/// `index`: read by coordinate. This IS the representable functor's index.
///
/// Given Field coordinates (x, y, z, w), converts x and y to integer indices
/// via nearest-neighbor (floor + clamp), looks up the buffer value, and
/// returns it as a Field.
impl Manifold<(Field, Field, Field, Field)> for DiscreteManifold {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: (Field, Field, Field, Field)) -> Field {
        let (x, y, _, _) = p;

        // If the buffer is empty, there's nothing to look up.
        // Fail loud: this is a programming error, not a recoverable condition.
        assert!(
            !self.buffer.is_empty(),
            "DiscreteManifold::eval called on empty buffer ({}x{})",
            self.width,
            self.height,
        );

        let zero = Field::from(0.0);
        let max_x = Field::from((self.width.saturating_sub(1)) as f32);
        let max_y = Field::from((self.height.saturating_sub(1)) as f32);

        // Nearest-neighbor: floor then clamp to valid range.
        let xi = x.floor().max(zero).min(max_x);
        let yi = y.floor().max(zero).min(max_y);

        // Linear index = floor(y) * width + floor(x)
        let w_field = Field::from(self.width as f32);
        let indices = yi.raw_mul(w_field).raw_add(xi);

        Field::gather(&self.buffer, indices)
    }
}

// ============================================================================
// FrameLattice: 2D grid (X and Y are loops, Z and W fixed)
// ============================================================================

/// A 2D frame lattice: X varies per pixel, Y per scanline.
/// Z and W are fixed (frame time and layer).
///
/// Iteration order: Y outer (scanlines), X inner (pixels).
/// X coordinates use `Field::sequential()` for SIMD lane alignment.
#[derive(Copy, Clone, Debug)]
pub struct FrameLattice {
    /// Width in pixels (X extent).
    pub width: usize,
    /// Height in scanlines (Y extent).
    pub height: usize,
    /// Fixed Z coordinate (typically frame time).
    pub z: f32,
    /// Fixed W coordinate (typically layer index, usually 0).
    pub w: f32,
}

/// Stored loop variable indices for FrameLattice.
const FRAME_LOOP_VARS: [u8; 2] = [0, 1]; // X=0, Y=1

impl FrameLattice {
    /// Convenience constructor: 2D frame with Z = time, W = 0.
    #[must_use]
    pub fn new(width: usize, height: usize, z: f32) -> Self {
        Self {
            width,
            height,
            z,
            w: 0.0,
        }
    }
}

impl LatticeDomain for FrameLattice {
    fn len(&self) -> usize {
        self.width * self.height
    }

    fn loop_vars(&self) -> &[u8] {
        &FRAME_LOOP_VARS
    }

    fn coord(&self, index: usize) -> (f32, f32, f32, f32) {
        assert!(
            index < self.len(),
            "FrameLattice::coord index {} out of bounds (len = {})",
            index,
            self.len(),
        );
        let x = (index % self.width) as f32;
        let y = (index / self.width) as f32;
        (x, y, self.z, self.w)
    }
}

impl Lattice for FrameLattice {
    fn collapse<M>(&self, manifold: &M) -> DiscreteManifold
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        let total = self.width * self.height;
        let mut buffer = vec![0.0f32; total];
        let mut packed = [0.0f32; PARALLELISM];

        let z_field = Field::from(self.z);
        let w_field = Field::from(self.w);
        let step = Field::from(PARALLELISM as f32);

        // Y outer, X inner (scanline order)
        for y in 0..self.height {
            let y_field = Field::from(y as f32);
            let row_offset = y * self.width;
            let mut x = 0usize;
            let mut x_field = Field::sequential(0.0);

            // SIMD hot path: full batches of PARALLELISM pixels
            while x + PARALLELISM <= self.width {
                let result = manifold.eval((x_field, y_field, z_field, w_field));
                result.store(&mut packed);
                buffer[row_offset + x..row_offset + x + PARALLELISM].copy_from_slice(&packed);

                x += PARALLELISM;
                x_field = x_field.raw_add(step);
            }

            // SIMD tail: evaluate the last partial batch
            if x < self.width {
                let result = manifold.eval((x_field, y_field, z_field, w_field));
                result.store(&mut packed);
                let tail_len = self.width - x;
                buffer[row_offset + x..row_offset + self.width]
                    .copy_from_slice(&packed[..tail_len]);
            }
        }

        DiscreteManifold {
            buffer,
            width: self.width,
            height: self.height,
        }
    }

    fn collapse_with<M>(&self, op: ReduceOp, manifold: &M) -> Field
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        let mut acc = op.identity();

        let z_field = Field::from(self.z);
        let w_field = Field::from(self.w);
        let step = Field::from(PARALLELISM as f32);

        for y in 0..self.height {
            let y_field = Field::from(y as f32);
            let mut x_field = Field::sequential(0.0);
            let mut x = 0usize;

            while x + PARALLELISM <= self.width {
                let result = manifold.eval((x_field, y_field, z_field, w_field));
                acc = op.apply(acc, result);

                x += PARALLELISM;
                x_field = x_field.raw_add(step);
            }

            // Tail: for reduction we still evaluate and fold, but only
            // the valid lanes contribute. We use the identity element for
            // out-of-bounds lanes so the fold is unaffected.
            if x < self.width {
                let result = manifold.eval((x_field, y_field, z_field, w_field));
                let tail_len = self.width - x;

                // Build a mask: lanes [0..tail_len) are valid, rest get identity.
                // We construct this by comparing sequential lane indices against tail_len.
                let lane_indices = Field::sequential(0.0);
                let threshold = Field::from(tail_len as f32);
                let mask = lane_indices.lt(threshold);
                // Select: valid lanes get result, invalid lanes get identity.
                let masked = Field::select_raw(mask, result, op.identity());
                acc = op.apply(acc, masked);
            }
        }

        acc
    }
}

// ============================================================================
// ScanlineLattice: 1D (only X varies)
// ============================================================================

/// A 1D scanline lattice: only X varies.
/// Y, Z, W are all fixed constants.
#[derive(Copy, Clone, Debug)]
pub struct ScanlineLattice {
    /// Width in pixels (X extent).
    pub width: usize,
    /// Fixed Y coordinate (scanline index).
    pub y: f32,
    /// Fixed Z coordinate.
    pub z: f32,
    /// Fixed W coordinate.
    pub w: f32,
}

/// Stored loop variable indices for ScanlineLattice.
const SCANLINE_LOOP_VARS: [u8; 1] = [0]; // X=0

impl ScanlineLattice {
    /// Convenience constructor.
    #[must_use]
    pub fn new(width: usize, y: f32, z: f32, w: f32) -> Self {
        Self { width, y, z, w }
    }
}

impl LatticeDomain for ScanlineLattice {
    fn len(&self) -> usize {
        self.width
    }

    fn loop_vars(&self) -> &[u8] {
        &SCANLINE_LOOP_VARS
    }

    fn coord(&self, index: usize) -> (f32, f32, f32, f32) {
        assert!(
            index < self.len(),
            "ScanlineLattice::coord index {} out of bounds (len = {})",
            index,
            self.len(),
        );
        (index as f32, self.y, self.z, self.w)
    }
}

impl Lattice for ScanlineLattice {
    fn collapse<M>(&self, manifold: &M) -> DiscreteManifold
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        let mut buffer = vec![0.0f32; self.width];
        let mut packed = [0.0f32; PARALLELISM];

        let y_field = Field::from(self.y);
        let z_field = Field::from(self.z);
        let w_field = Field::from(self.w);
        let step = Field::from(PARALLELISM as f32);
        let mut x = 0usize;
        let mut x_field = Field::sequential(0.0);

        while x + PARALLELISM <= self.width {
            let result = manifold.eval((x_field, y_field, z_field, w_field));
            result.store(&mut packed);
            buffer[x..x + PARALLELISM].copy_from_slice(&packed);

            x += PARALLELISM;
            x_field = x_field.raw_add(step);
        }

        if x < self.width {
            let result = manifold.eval((x_field, y_field, z_field, w_field));
            result.store(&mut packed);
            let tail_len = self.width - x;
            buffer[x..self.width].copy_from_slice(&packed[..tail_len]);
        }

        DiscreteManifold {
            buffer,
            width: self.width,
            height: 1,
        }
    }

    fn collapse_with<M>(&self, op: ReduceOp, manifold: &M) -> Field
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        let mut acc = op.identity();

        let y_field = Field::from(self.y);
        let z_field = Field::from(self.z);
        let w_field = Field::from(self.w);
        let step = Field::from(PARALLELISM as f32);
        let mut x = 0usize;
        let mut x_field = Field::sequential(0.0);

        while x + PARALLELISM <= self.width {
            let result = manifold.eval((x_field, y_field, z_field, w_field));
            acc = op.apply(acc, result);

            x += PARALLELISM;
            x_field = x_field.raw_add(step);
        }

        if x < self.width {
            let result = manifold.eval((x_field, y_field, z_field, w_field));
            let tail_len = self.width - x;
            let lane_indices = Field::sequential(0.0);
            let threshold = Field::from(tail_len as f32);
            let mask = lane_indices.lt(threshold);
            let masked = Field::select_raw(mask, result, op.identity());
            acc = op.apply(acc, masked);
        }

        acc
    }
}

// ============================================================================
// PointLattice: all fixed, degenerate case (0D)
// ============================================================================

/// A degenerate lattice: all coordinates fixed. Single point.
///
/// This is `Lattice<1,1,1,1>` -- the zero-dimensional case.
/// No loops, just evaluate once and wrap the result.
#[derive(Copy, Clone, Debug)]
pub struct PointLattice {
    /// Fixed X coordinate.
    pub x: f32,
    /// Fixed Y coordinate.
    pub y: f32,
    /// Fixed Z coordinate.
    pub z: f32,
    /// Fixed W coordinate.
    pub w: f32,
}

/// Stored loop variable indices for PointLattice (empty: no loops).
const POINT_LOOP_VARS: [u8; 0] = [];

impl PointLattice {
    /// Create a point lattice at the given coordinates.
    #[must_use]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
}

impl LatticeDomain for PointLattice {
    fn len(&self) -> usize {
        1
    }

    fn loop_vars(&self) -> &[u8] {
        &POINT_LOOP_VARS
    }

    fn coord(&self, index: usize) -> (f32, f32, f32, f32) {
        assert!(
            index == 0,
            "PointLattice::coord index {} out of bounds (len = 1)",
            index,
        );
        (self.x, self.y, self.z, self.w)
    }
}

impl Lattice for PointLattice {
    fn collapse<M>(&self, manifold: &M) -> DiscreteManifold
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        let result = manifold.eval((
            Field::from(self.x),
            Field::from(self.y),
            Field::from(self.z),
            Field::from(self.w),
        ));

        // Extract the first lane (all lanes are identical since all coords are broadcast).
        let mut packed = [0.0f32; PARALLELISM];
        result.store(&mut packed);

        DiscreteManifold {
            buffer: vec![packed[0]],
            width: 1,
            height: 1,
        }
    }

    fn collapse_with<M>(&self, _op: ReduceOp, manifold: &M) -> Field
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        // Single point: the fold of one element is itself, regardless of op.
        manifold.eval((
            Field::from(self.x),
            Field::from(self.y),
            Field::from(self.z),
            Field::from(self.w),
        ))
    }
}
#[cfg(test)]
mod tests;
