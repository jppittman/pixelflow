//! # Lattice: Representable Functor for Manifold Evaluation
//!
//! A Lattice is a finite box domain that collapses a Manifold into a discrete
//! buffer. This is the `tabulate`/`index` pair from representable functors:
//!
//! - **`collapse`** = `tabulate`: `(Rep -> a) -> F a` -- evaluate at every point
//! - **`DiscreteManifold::eval`** = `index`: `F a -> Rep -> a` -- read back by coordinate
//! - **Isomorphism**: `index(collapse(f, domain), i) = f(coord(i))` (up to discretization)
//!
//! Nothing computes until a Lattice demands it. A single-point evaluation is
//! just `Lattice::point` -- the degenerate case with all coordinates fixed.
//!
//! There is one `Lattice` type, not one per shape. An axis with extent 1 is
//! fixed at its origin; an axis with extent > 1 is a loop dimension. The
//! constructors (`frame`, `scanline`, `point`, `index`, `index2`) are sugar
//! for common shapes -- the shape is data, not a type. Extents only need to
//! be static at JIT-compile time, which is when the kernel is specialized.
//!
//! The naive implementations here iterate and call `manifold.eval()` per-batch.
//! The JIT-backed fast path will override `collapse` later, reading the DAG,
//! computing variance, and emitting nested loops with hoisting.

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

    /// Fold all SIMD lanes of `acc` into a single scalar.
    #[inline]
    pub(crate) fn horizontal(self, acc: Field) -> f32 {
        let mut lanes = [0.0f32; PARALLELISM];
        acc.store(&mut lanes);
        match self {
            ReduceOp::Add => lanes.iter().sum(),
            ReduceOp::Mul => lanes.iter().product(),
            ReduceOp::Min => lanes.iter().copied().fold(f32::INFINITY, f32::min),
            ReduceOp::Max => lanes.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        }
    }
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
// Lattice: a finite box domain over the four coordinate axes
// ============================================================================

/// A finite box domain over the four coordinate axes (X, Y, Z, W).
///
/// `extent[i]` is the number of samples along axis `i`; an axis with extent 1
/// is fixed at `origin[i]`. `origin[i]` is the coordinate of index 0 on each
/// axis. Iteration is row-major with X innermost (SIMD lanes ride X).
///
/// The shape is data, not a type: a frame, a scanline, a point, and a tensor
/// index range are all the same `Lattice` with different extents. The JIT
/// specializes on the extents at kernel-compile time.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Lattice {
    /// Samples per axis: `[x, y, z, w]`. Extent 1 = fixed axis.
    pub extent: [u32; 4],
    /// Coordinate of index 0 on each axis.
    pub origin: [f32; 4],
}

impl Lattice {
    /// A 2D pixel frame: X varies per pixel, Y per scanline; Z is the fixed
    /// frame time, W is fixed at 0.
    #[must_use]
    pub fn frame(width: usize, height: usize, z: f32) -> Self {
        Self {
            extent: [width as u32, height as u32, 1, 1],
            origin: [0.0, 0.0, z, 0.0],
        }
    }

    /// A 1D scanline: only X varies; Y, Z, W are fixed.
    #[must_use]
    pub fn scanline(width: usize, y: f32, z: f32, w: f32) -> Self {
        Self {
            extent: [width as u32, 1, 1, 1],
            origin: [0.0, y, z, w],
        }
    }

    /// A single point: all coordinates fixed. The degenerate (0-loop) case.
    #[must_use]
    pub fn point(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self {
            extent: [1, 1, 1, 1],
            origin: [x, y, z, w],
        }
    }

    /// A 1D index range `[0, len)` over X. Feature indices, not pixels.
    #[must_use]
    pub fn index(len: usize) -> Self {
        Self {
            extent: [len as u32, 1, 1, 1],
            origin: [0.0; 4],
        }
    }

    /// A 2D index range `[0, width) x [0, height)` over X and Y.
    /// Weight-matrix indices: X = input dim, Y = output dim.
    #[must_use]
    pub fn index2(width: usize, height: usize) -> Self {
        Self {
            extent: [width as u32, height as u32, 1, 1],
            origin: [0.0; 4],
        }
    }

    // ───────────────────── domain queries ──────────────────────

    /// Number of points in this domain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.extent.iter().map(|&e| e as usize).product()
    }

    /// Whether the domain is empty (any axis has extent 0).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bitmask of loop axes (extent > 1): bit 0 = X, 1 = Y, 2 = Z, 3 = W.
    /// Fixed axes are constants from the kernel's point of view.
    #[must_use]
    pub fn loop_mask(&self) -> u8 {
        let mut mask = 0u8;
        for (i, &e) in self.extent.iter().enumerate() {
            if e > 1 {
                mask |= 1 << i;
            }
        }
        mask
    }

    /// Map a linear index to concrete coordinate values (X fastest).
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    #[must_use]
    pub fn coord(&self, index: usize) -> (f32, f32, f32, f32) {
        assert!(
            index < self.len(),
            "Lattice::coord index {} out of bounds (len = {})",
            index,
            self.len(),
        );
        let [ex, ey, ez, _] = self.extent.map(|e| e as usize);
        let x = index % ex;
        let rest = index / ex;
        let y = rest % ey;
        let rest = rest / ey;
        let z = rest % ez;
        let w = rest / ez;
        (
            self.origin[0] + x as f32,
            self.origin[1] + y as f32,
            self.origin[2] + z as f32,
            self.origin[3] + w as f32,
        )
    }

    // ───────────────────── collapse (tabulate) ──────────────────────

    /// Collapse: evaluate the manifold at every point in the domain.
    /// Returns a discrete manifold (buffer lookup) with `width = extent[0]`
    /// and `height` = the product of the remaining extents.
    ///
    /// This is `tabulate`: `(Rep -> a) -> F a`.
    pub fn collapse<M>(&self, manifold: &M) -> DiscreteManifold
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        let [ex, ey, ez, ew] = self.extent.map(|e| e as usize);
        let mut buffer = vec![0.0f32; self.len()];
        let mut packed = [0.0f32; PARALLELISM];
        let step = Field::from(PARALLELISM as f32);

        let mut row = 0usize;
        for w in 0..ew {
            let w_field = Field::from(self.origin[3] + w as f32);
            for z in 0..ez {
                let z_field = Field::from(self.origin[2] + z as f32);
                for y in 0..ey {
                    let y_field = Field::from(self.origin[1] + y as f32);
                    let row_offset = row * ex;
                    let mut x = 0usize;
                    let mut x_field = Field::sequential(self.origin[0]);

                    // SIMD hot path: full batches of PARALLELISM points.
                    while x + PARALLELISM <= ex {
                        let result = manifold.eval((x_field, y_field, z_field, w_field));
                        result.store(&mut packed);
                        buffer[row_offset + x..row_offset + x + PARALLELISM]
                            .copy_from_slice(&packed);
                        x += PARALLELISM;
                        x_field = x_field.raw_add(step);
                    }

                    // SIMD tail: evaluate the last partial batch.
                    if x < ex {
                        let result = manifold.eval((x_field, y_field, z_field, w_field));
                        result.store(&mut packed);
                        let tail_len = ex - x;
                        buffer[row_offset + x..row_offset + ex]
                            .copy_from_slice(&packed[..tail_len]);
                    }
                    row += 1;
                }
            }
        }

        DiscreteManifold {
            buffer,
            width: ex,
            height: ey * ez * ew,
        }
    }

    /// Bake a [`Kernel`](pixelflow_ir::Kernel) — the front-end value — over the
    /// domain: JIT-compile it once (through the global cache) and tabulate. The
    /// JIT-first path: no combinator manifold, no `Lower`, just the arena the
    /// `Kernel` already carries. Its `Dwrt` derivatives are resolved by the
    /// compiler during codegen. Falls back to nothing — a `Kernel` is always
    /// an arena, always compilable — except when this build's `Field` width is
    /// not the JIT's, where it panics rather than silently mis-tabulating.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[must_use]
    pub fn bake(&self, kernel: &pixelflow_ir::Kernel) -> DiscreteManifold {
        assert_eq!(
            core::mem::size_of::<Field>(),
            pixelflow_ir::JIT_VECTOR_BYTES,
            "Lattice::bake: Field width does not match the JIT's emitted width"
        );
        let (arena, root) = kernel.parts();
        let jit = pixelflow_ir::jit_cache::compile_cached(arena, root)
            .expect("kernel failed to compile");
        self.collapse(&RealizedKernel(jit))
    }

    /// Fold all points of the lattice into a per-lane SIMD accumulator.
    ///
    /// Each SIMD lane folds an independent stripe of X; the lanes are NOT
    /// combined. Use [`Lattice::collapse_scalar`] when you want one number
    /// for the whole domain.
    pub fn collapse_with<M>(&self, op: ReduceOp, manifold: &M) -> Field
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        self.fold_lanes(op, manifold)
    }

    /// Fold all points of the lattice into a single scalar using `op`.
    ///
    /// This is the full horizontal reduction: SIMD lanes are folded together
    /// at the end. Use this for index-space semantics (dot products, losses).
    pub fn collapse_scalar<M>(&self, op: ReduceOp, manifold: &M) -> f32
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        op.horizontal(self.fold_lanes(op, manifold))
    }

    /// Shared lane-wise fold over the whole domain (tail lanes masked to the
    /// monoid identity so they do not contribute).
    fn fold_lanes<M>(&self, op: ReduceOp, manifold: &M) -> Field
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        let [ex, ey, ez, ew] = self.extent.map(|e| e as usize);
        let mut acc = op.identity();
        let step = Field::from(PARALLELISM as f32);

        for w in 0..ew {
            let w_field = Field::from(self.origin[3] + w as f32);
            for z in 0..ez {
                let z_field = Field::from(self.origin[2] + z as f32);
                for y in 0..ey {
                    let y_field = Field::from(self.origin[1] + y as f32);
                    let mut x = 0usize;
                    let mut x_field = Field::sequential(self.origin[0]);

                    while x + PARALLELISM <= ex {
                        let result = manifold.eval((x_field, y_field, z_field, w_field));
                        acc = op.apply(acc, result);
                        x += PARALLELISM;
                        x_field = x_field.raw_add(step);
                    }

                    // Tail: only the valid lanes contribute; out-of-bounds
                    // lanes get the identity element so the fold is unaffected.
                    if x < ex {
                        let result = manifold.eval((x_field, y_field, z_field, w_field));
                        let tail_len = ex - x;
                        let lane_indices = Field::sequential(0.0);
                        let threshold = Field::from(tail_len as f32);
                        let mask = lane_indices.lt(threshold);
                        let masked = Field::select_raw(mask, result, op.identity());
                        acc = op.apply(acc, masked);
                    }
                }
            }
        }

        acc
    }

    // ───────────────────── partial reduction (matmul primitive) ─────────────

    /// Reduce one axis of a 2D (X/Y) lattice, returning a 1D `DiscreteManifold`.
    ///
    /// The surviving dimension is remapped to X in the output (height = 1).
    ///
    /// - `axis = 0`: reduce over X, keep Y. Output `[j]` = fold over `i` of `m(i, j)`.
    /// - `axis = 1`: reduce over Y, keep X. Output `[i]` = fold over `j` of `m(i, j)`.
    ///
    /// This is the matmul primitive: `collapse_axis(0, Add, W * input)` computes
    /// `output[j] = sum_i W(i,j) * input(i)`.
    ///
    /// # Panics
    ///
    /// Panics if `axis` is not 0 or 1, or if the Z/W axes are not fixed.
    pub fn collapse_axis<M>(&self, axis: usize, op: ReduceOp, manifold: &M) -> DiscreteManifold
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        assert!(
            self.extent[2] <= 1 && self.extent[3] <= 1,
            "Lattice::collapse_axis requires fixed Z and W axes (extents {:?})",
            self.extent,
        );
        match axis {
            0 => self.collapse_axis0(op, manifold),
            1 => self.collapse_axis1(op, manifold),
            _ => panic!(
                "Lattice::collapse_axis: axis {} out of range (must be 0 or 1)",
                axis
            ),
        }
    }

    /// Reduce over X, produce one value per Y. Output indexed by X (height=1).
    fn collapse_axis0<M>(&self, op: ReduceOp, manifold: &M) -> DiscreteManifold
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        let ex = self.extent[0] as usize;
        let out_len = self.extent[1] as usize;
        let mut out_buffer = vec![0.0f32; out_len];

        let z_field = Field::from(self.origin[2]);
        let w_field = Field::from(self.origin[3]);
        let step = Field::from(PARALLELISM as f32);

        for (j, out) in out_buffer.iter_mut().enumerate() {
            let y_field = Field::from(self.origin[1] + j as f32);
            let mut acc = op.identity();

            let mut i = 0usize;
            let mut x_field = Field::sequential(self.origin[0]);

            while i + PARALLELISM <= ex {
                let val = manifold.eval((x_field, y_field, z_field, w_field));
                acc = op.apply(acc, val);
                i += PARALLELISM;
                x_field = x_field.raw_add(step);
            }

            if i < ex {
                let val = manifold.eval((x_field, y_field, z_field, w_field));
                let tail_len = ex - i;
                let lane_indices = Field::sequential(0.0);
                let threshold = Field::from(tail_len as f32);
                let mask = lane_indices.lt(threshold);
                let masked = Field::select_raw(mask, val, op.identity());
                acc = op.apply(acc, masked);
            }

            *out = op.horizontal(acc);
        }

        DiscreteManifold::new(out_buffer, out_len, 1)
    }

    /// Reduce over Y, produce one value per X. Output indexed by X (height=1).
    fn collapse_axis1<M>(&self, op: ReduceOp, manifold: &M) -> DiscreteManifold
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        let ey = self.extent[1] as usize;
        let out_len = self.extent[0] as usize;
        let mut out_buffer = vec![0.0f32; out_len];

        let z_field = Field::from(self.origin[2]);
        let w_field = Field::from(self.origin[3]);

        for (i, out) in out_buffer.iter_mut().enumerate() {
            let x_field = Field::from(self.origin[0] + i as f32);
            let mut acc = op.identity();

            for j in 0..ey {
                let y_field = Field::from(self.origin[1] + j as f32);
                let val = manifold.eval((x_field, y_field, z_field, w_field));
                acc = op.apply(acc, val);
            }

            // All SIMD lanes are uniform (scalar broadcast over a fixed X index).
            // A cross-lane reduction would be wrong here: for Add it would multiply
            // the result by PARALLELISM, for Mul it would raise to the PARALLELISM-th
            // power, etc. Lane 0 is always correct when all lanes carry the same value.
            let mut packed = [0.0f32; PARALLELISM];
            acc.store(&mut packed);
            #[cfg(debug_assertions)]
            for k in 1..PARALLELISM {
                debug_assert!(
                    (packed[k] - packed[0]).abs() < 1e-4,
                    "collapse_axis1: lane {} ({}) differs from lane 0 ({}) — manifold must be lane-uniform",
                    k,
                    packed[k],
                    packed[0],
                );
            }
            *out = packed[0];
        }

        DiscreteManifold::new(out_buffer, out_len, 1)
    }
}

#[cfg(test)]
mod tests;

/// A JIT-compiled kernel driving the generic collapse loop — the fast path
/// of [`Lattice::bake`]. `Field` is transmuted to the emitter's vector ABI;
/// `bake` guards the width match before constructing one.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
struct RealizedKernel(alloc::sync::Arc<pixelflow_ir::JitManifold>);

/// The vector type of the JIT's call ABI on this build.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
type JitVec = core::arch::x86_64::__m128;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
type JitVec = core::arch::x86_64::__m512;
#[cfg(target_arch = "aarch64")]
type JitVec = core::arch::aarch64::float32x4_t;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl Manifold<(Field, Field, Field, Field)> for RealizedKernel {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, (x, y, z, w): (Field, Field, Field, Field)) -> Field {
        // SAFETY: realize checked size_of::<Field>() == JIT_VECTOR_BYTES, and
        // the code was emitted by our own backend for exactly that ABI.
        unsafe {
            core::mem::transmute::<JitVec, Field>(self.0.call(
                core::mem::transmute::<Field, JitVec>(x),
                core::mem::transmute::<Field, JitVec>(y),
                core::mem::transmute::<Field, JitVec>(z),
                core::mem::transmute::<Field, JitVec>(w),
            ))
        }
    }
}
