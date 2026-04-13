use crate::numeric::Numeric;
use crate::{Field, Manifold, PARALLELISM};
use alloc::vec;
use super::{DiscreteManifold, Lattice, LatticeDomain, ReduceOp};

// ============================================================================
// IndexLattice1D: 1D index domain (feature/tensor indexing, not pixel space)
// ============================================================================

/// A 1D index lattice: X varies over [0, len), Y/Z/W fixed at 0.
///
/// Semantics: feature indices, not pixel coordinates.
/// Use this for ML tensor operations: matmul inputs, layer outputs, etc.
#[derive(Copy, Clone, Debug)]
pub struct IndexLattice1D {
    /// Number of indices (X extent).
    pub len: usize,
}

const INDEX1D_LOOP_VARS: [u8; 1] = [0]; // X=0 only

impl IndexLattice1D {
    /// Create a 1D index lattice over [0, len).
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl LatticeDomain for IndexLattice1D {
    fn len(&self) -> usize {
        self.len
    }

    fn loop_vars(&self) -> &[u8] {
        &INDEX1D_LOOP_VARS
    }

    fn coord(&self, index: usize) -> (f32, f32, f32, f32) {
        assert!(
            index < self.len,
            "IndexLattice1D::coord index {} out of bounds (len = {})",
            index,
            self.len,
        );
        (index as f32, 0.0, 0.0, 0.0)
    }
}

impl Lattice for IndexLattice1D {
    fn collapse<M>(&self, manifold: &M) -> DiscreteManifold
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        let mut buffer = vec![0.0f32; self.len];
        let mut packed = [0.0f32; PARALLELISM];

        let y_field = Field::from(0.0);
        let z_field = Field::from(0.0);
        let w_field = Field::from(0.0);
        let step = Field::from(PARALLELISM as f32);

        let mut x = 0usize;
        let mut x_field = Field::sequential(0.0);

        while x + PARALLELISM <= self.len {
            let result = manifold.eval((x_field, y_field, z_field, w_field));
            result.store(&mut packed);
            buffer[x..x + PARALLELISM].copy_from_slice(&packed);
            x += PARALLELISM;
            x_field = x_field.raw_add(step);
        }

        if x < self.len {
            let result = manifold.eval((x_field, y_field, z_field, w_field));
            result.store(&mut packed);
            let tail_len = self.len - x;
            buffer[x..self.len].copy_from_slice(&packed[..tail_len]);
        }

        DiscreteManifold {
            buffer,
            width: self.len,
            height: 1,
        }
    }

    /// Reduce all indices in [0, len) to a single scalar using `op`.
    ///
    /// Performs a SIMD fold over all indices, then horizontally reduces all
    /// SIMD lanes to a single scalar value. The returned `Field` has all
    /// lanes set to that scalar (via `Field::from(scalar)`).
    ///
    /// This differs from pixel-space `collapse_with` impls which return a
    /// per-lane accumulator. Index-space semantics: one result for the whole range.
    fn collapse_with<M>(&self, op: ReduceOp, manifold: &M) -> Field
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        let mut acc = op.identity();

        let y_field = Field::from(0.0);
        let z_field = Field::from(0.0);
        let w_field = Field::from(0.0);
        let step = Field::from(PARALLELISM as f32);

        let mut x = 0usize;
        let mut x_field = Field::sequential(0.0);

        while x + PARALLELISM <= self.len {
            let result = manifold.eval((x_field, y_field, z_field, w_field));
            acc = op.apply(acc, result);
            x += PARALLELISM;
            x_field = x_field.raw_add(step);
        }

        if x < self.len {
            let result = manifold.eval((x_field, y_field, z_field, w_field));
            let tail_len = self.len - x;
            let lane_indices = Field::sequential(0.0);
            let threshold = Field::from(tail_len as f32);
            let mask = lane_indices.lt(threshold);
            let masked = Field::select_raw(mask, result, op.identity());
            acc = op.apply(acc, masked);
        }

        // Horizontal reduction: fold all SIMD lanes to a single scalar.
        // collapse_with reduces the entire 1D lattice to one value.
        let mut lanes = [0.0f32; PARALLELISM];
        acc.store(&mut lanes);
        let scalar: f32 = match op {
            ReduceOp::Add => lanes.iter().sum(),
            ReduceOp::Mul => lanes.iter().product(),
            ReduceOp::Min => lanes.iter().cloned().fold(f32::INFINITY, f32::min),
            ReduceOp::Max => lanes.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        };
        Field::from(scalar)
    }
}

// ============================================================================
// IndexLattice2D: 2D index domain (weight matrix / tensor indexing)
// ============================================================================

/// A 2D index lattice: X varies over [0, width), Y varies over [0, height).
/// Z and W fixed at 0.
///
/// Semantics: weight matrix indices (X = input dim, Y = output dim).
/// Use with `collapse_axis` to express matmul.
#[derive(Copy, Clone, Debug)]
pub struct IndexLattice2D {
    /// X extent (e.g. input dimension).
    pub width: usize,
    /// Y extent (e.g. output dimension).
    pub height: usize,
}

const INDEX2D_LOOP_VARS: [u8; 2] = [0, 1]; // X=0, Y=1

impl IndexLattice2D {
    /// Create a 2D index lattice over [0, width) × [0, height).
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }
}

impl LatticeDomain for IndexLattice2D {
    fn len(&self) -> usize {
        self.width * self.height
    }

    fn loop_vars(&self) -> &[u8] {
        &INDEX2D_LOOP_VARS
    }

    fn coord(&self, index: usize) -> (f32, f32, f32, f32) {
        assert!(
            index < self.len(),
            "IndexLattice2D::coord index {} out of bounds (len = {})",
            index,
            self.len(),
        );
        let x = (index % self.width) as f32;
        let y = (index / self.width) as f32;
        (x, y, 0.0, 0.0)
    }
}

impl Lattice for IndexLattice2D {
    fn collapse<M>(&self, manifold: &M) -> DiscreteManifold
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        let total = self.width * self.height;
        let mut buffer = vec![0.0f32; total];
        let mut packed = [0.0f32; PARALLELISM];

        let z_field = Field::from(0.0);
        let w_field = Field::from(0.0);
        let step = Field::from(PARALLELISM as f32);

        // Y outer, X inner (matches row-major layout: buffer[y * width + x])
        for y in 0..self.height {
            let y_field = Field::from(y as f32);
            let row_offset = y * self.width;
            let mut x = 0usize;
            let mut x_field = Field::sequential(0.0);

            while x + PARALLELISM <= self.width {
                let result = manifold.eval((x_field, y_field, z_field, w_field));
                result.store(&mut packed);
                buffer[row_offset + x..row_offset + x + PARALLELISM].copy_from_slice(&packed);
                x += PARALLELISM;
                x_field = x_field.raw_add(step);
            }

            if x < self.width {
                let result = manifold.eval((x_field, y_field, z_field, w_field));
                result.store(&mut packed);
                let tail_len = self.width - x;
                buffer[row_offset + x..row_offset + self.width].copy_from_slice(&packed[..tail_len]);
            }
        }

        DiscreteManifold::new(buffer, self.width, self.height)
    }

    /// Reduce all indices in [0, width) × [0, height) to a single scalar using `op`.
    ///
    /// Performs a SIMD fold over all indices (Y outer, X inner), then horizontally
    /// reduces all SIMD lanes to a single scalar value. The returned `Field` has all
    /// lanes set to that scalar (via `Field::from(scalar)`).
    fn collapse_with<M>(&self, op: ReduceOp, manifold: &M) -> Field
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        let mut acc = op.identity();

        let z_field = Field::from(0.0);
        let w_field = Field::from(0.0);
        let step = Field::from(PARALLELISM as f32);

        for y in 0..self.height {
            let y_field = Field::from(y as f32);
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
        }

        // Horizontal reduction: fold all SIMD lanes to a single scalar.
        // collapse_with reduces the entire 2D lattice to one value.
        let mut lanes = [0.0f32; PARALLELISM];
        acc.store(&mut lanes);
        let scalar: f32 = match op {
            ReduceOp::Add => lanes.iter().sum(),
            ReduceOp::Mul => lanes.iter().product(),
            ReduceOp::Min => lanes.iter().cloned().fold(f32::INFINITY, f32::min),
            ReduceOp::Max => lanes.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        };
        Field::from(scalar)
    }
}

// ============================================================================
// IndexLattice2D — partial reduction (matmul primitive)
// ============================================================================

impl IndexLattice2D {
    /// Reduce one axis of a 2D manifold, returning a 1D `DiscreteManifold`.
    ///
    /// The surviving dimension is remapped to X in the output (height = 1).
    ///
    /// - `axis = 0`: reduce over X (width), keep Y. Output `[j]` = fold over `i` of `m(i, j)`.
    /// - `axis = 1`: reduce over Y (height), keep X. Output `[i]` = fold over `j` of `m(i, j)`.
    ///
    /// This is the matmul primitive: `collapse_axis(0, Add, W * input)` computes
    /// `output[j] = Σᵢ W(i,j) * input(i)`.
    ///
    /// # Panics
    ///
    /// Panics if `axis` is not 0 or 1.
    pub fn collapse_axis<M>(&self, axis: usize, op: ReduceOp, manifold: &M) -> DiscreteManifold
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        match axis {
            0 => self.collapse_axis0(op, manifold),
            1 => self.collapse_axis1(op, manifold),
            _ => panic!(
                "IndexLattice2D::collapse_axis: axis {} out of range (must be 0 or 1)",
                axis
            ),
        }
    }

    /// Reduce over X, produce one value per Y. Output indexed by X (height=1).
    fn collapse_axis0<M>(&self, op: ReduceOp, manifold: &M) -> DiscreteManifold
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        // For each output index j (Y), fold over i (X).
        // Result: width = self.height (j is the surviving dim), height = 1.
        let out_len = self.height;
        let mut out_buffer = vec![0.0f32; out_len];

        let z_field = Field::from(0.0);
        let w_field = Field::from(0.0);
        let step = Field::from(PARALLELISM as f32);

        for j in 0..self.height {
            let y_field = Field::from(j as f32);
            let mut acc = op.identity();

            let mut i = 0usize;
            let mut x_field = Field::sequential(0.0);

            while i + PARALLELISM <= self.width {
                let val = manifold.eval((x_field, y_field, z_field, w_field));
                acc = op.apply(acc, val);
                i += PARALLELISM;
                x_field = x_field.raw_add(step);
            }

            if i < self.width {
                let val = manifold.eval((x_field, y_field, z_field, w_field));
                let tail_len = self.width - i;
                let lane_indices = Field::sequential(0.0);
                let threshold = Field::from(tail_len as f32);
                let mask = lane_indices.lt(threshold);
                let masked = Field::select_raw(mask, val, op.identity());
                acc = op.apply(acc, masked);
            }

            // Horizontal reduction: sum all SIMD lanes into scalar.
            let mut packed = [0.0f32; PARALLELISM];
            acc.store(&mut packed);
            let scalar: f32 = match op {
                ReduceOp::Add => packed.iter().sum(),
                ReduceOp::Mul => packed.iter().product(),
                ReduceOp::Min => packed.iter().cloned().fold(f32::INFINITY, f32::min),
                ReduceOp::Max => packed.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            };
            out_buffer[j] = scalar;
        }

        DiscreteManifold::new(out_buffer, out_len, 1)
    }

    /// Reduce over Y, produce one value per X. Output indexed by X (height=1).
    fn collapse_axis1<M>(&self, op: ReduceOp, manifold: &M) -> DiscreteManifold
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        // For each output index i (X), fold over j (Y).
        // Result: width = self.width (i is the surviving dim), height = 1.
        let out_len = self.width;
        let mut out_buffer = vec![0.0f32; out_len];

        let z_field = Field::from(0.0);
        let w_field = Field::from(0.0);

        for i in 0..self.width {
            let x_field = Field::from(i as f32);
            let mut acc = op.identity();

            for j in 0..self.height {
                let y_field = Field::from(j as f32);
                let val = manifold.eval((x_field, y_field, z_field, w_field));
                acc = op.apply(acc, val);
            }

            // All SIMD lanes are uniform (scalar broadcast over a fixed X index).
            // Assert uniformity in debug mode, then extract lane 0 as the scalar.
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
            out_buffer[i] = packed[0];
        }

        DiscreteManifold::new(out_buffer, out_len, 1)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PARALLELISM;
    use crate::variables::X;

    #[test]
    fn index_lattice1d_collapse_identity() {
        let lattice = IndexLattice1D::new(4);
        let result = lattice.collapse(&X);
        assert_eq!(result.width(), 4);
        assert_eq!(result.height(), 1);
        let buf = result.buffer();
        assert!((buf[0] - 0.0).abs() < 1e-6);
        assert!((buf[1] - 1.0).abs() < 1e-6);
        assert!((buf[2] - 2.0).abs() < 1e-6);
        assert!((buf[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn index_lattice1d_collapse_with_sum() {
        // Sum of [0,1,2,3] across the whole lattice = 6.
        let lattice = IndexLattice1D::new(4);
        let result = lattice.collapse_with(ReduceOp::Add, &X);
        let mut packed = [0.0f32; PARALLELISM];
        result.store(&mut packed);
        // collapse_with must horizontal-reduce all SIMD lanes to a single scalar,
        // so all lanes should equal 6.0.
        assert!((packed[0] - 6.0).abs() < 1e-5, "expected 6.0, got {}", packed[0]);
    }

    #[test]
    fn index_lattice2d_collapse_xy_sum() {
        use crate::variables::Y;
        // 3×2 lattice (width=3, height=2). Values = X + Y.
        // Row 0 (Y=0): [0,1,2]. Row 1 (Y=1): [1,2,3].
        let lattice = IndexLattice2D::new(3, 2);
        let result = lattice.collapse(&(X + Y));
        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 2);
        let buf = result.buffer();
        // Row-major: [y=0,x=0],[y=0,x=1],[y=0,x=2],[y=1,x=0],[y=1,x=1],[y=1,x=2]
        assert!((buf[0] - 0.0).abs() < 1e-6); // x=0, y=0
        assert!((buf[1] - 1.0).abs() < 1e-6); // x=1, y=0
        assert!((buf[2] - 2.0).abs() < 1e-6); // x=2, y=0
        assert!((buf[3] - 1.0).abs() < 1e-6); // x=0, y=1
        assert!((buf[4] - 2.0).abs() < 1e-6); // x=1, y=1
        assert!((buf[5] - 3.0).abs() < 1e-6); // x=2, y=1
    }

    #[test]
    fn index_lattice2d_collapse_with_sum() {
        use crate::variables::{X, Y};
        // 3×2 lattice. Values = X + Y.
        // Sum = (0+0) + (1+0) + (2+0) + (0+1) + (1+1) + (2+1) = 0+1+2+1+2+3 = 9.
        let lattice = IndexLattice2D::new(3, 2);
        let result = lattice.collapse_with(ReduceOp::Add, &(X + Y));
        let mut packed = [0.0f32; PARALLELISM];
        result.store(&mut packed);
        // All lanes must equal 9.0 (horizontal reduction, not per-lane accumulation).
        assert!(
            (packed[0] - 9.0).abs() < 1e-5,
            "expected 9.0, got {}",
            packed[0]
        );
        assert!(
            (packed[1] - 9.0).abs() < 1e-5,
            "lane 1 not uniform — collapse_with must be a full horizontal reduction"
        );
    }

    #[test]
    fn collapse_axis0_dot_product() {
        use crate::{Field, Manifold};
        use crate::lattice::DiscreteManifold;

        // W = column-major layout for matmul:
        // W(input_i=X, output_j=Y): W(0,0)=1, W(1,0)=3, W(0,1)=2, W(1,1)=4
        // Row-major buffer (Y outer, X inner): [W(0,0), W(1,0), W(0,1), W(1,1)] = [1, 3, 2, 4]
        let w_buf = vec![1.0f32, 3.0, 2.0, 4.0];
        let w = DiscreteManifold::new(w_buf, 2, 2);
        let x_buf = vec![1.0f32, 2.0];
        let x_vec = DiscreteManifold::new(x_buf, 2, 1);

        struct Product {
            w: DiscreteManifold,
            x: DiscreteManifold,
        }
        impl Manifold<(Field, Field, Field, Field)> for Product {
            type Output = Field;
            fn eval(&self, (xi, yj, _, _): (Field, Field, Field, Field)) -> Field {
                let zero = Field::from(0.0);
                let w_val = self.w.eval((xi, yj, zero, zero));
                let x_val = self.x.eval((xi, zero, zero, zero));
                (w_val * x_val).eval((xi, yj, zero, zero))
            }
        }

        let lattice = IndexLattice2D::new(2, 2); // width=INPUT=2, height=OUTPUT=2
        let result = lattice.collapse_axis(0, ReduceOp::Add, &Product { w, x: x_vec });
        // result: width=2 (= self.height), height=1
        // result[0] = W(0,0)*x(0) + W(1,0)*x(1) = 1*1 + 3*2 = 7
        // result[1] = W(0,1)*x(0) + W(1,1)*x(1) = 2*1 + 4*2 = 10
        assert_eq!(result.width(), 2);
        assert_eq!(result.height(), 1);
        let buf = result.buffer();
        assert!((buf[0] - 7.0).abs() < 1e-4, "expected 7.0, got {}", buf[0]);
        assert!((buf[1] - 10.0).abs() < 1e-4, "expected 10.0, got {}", buf[1]);
    }

    #[test]
    fn collapse_axis1_row_sum() {
        use crate::variables::Y;
        // 2×3 lattice (width=2, height=3). Values = Y.
        // collapse_axis(1, Add): for each X=i, sum Y over [0,3) = 0+1+2 = 3
        // result: width=2, height=1. result[0]=3, result[1]=3.
        let lattice = IndexLattice2D::new(2, 3);
        let result = lattice.collapse_axis(1, ReduceOp::Add, &Y);
        assert_eq!(result.width(), 2);
        assert_eq!(result.height(), 1);
        let buf = result.buffer();
        assert!((buf[0] - 3.0).abs() < 1e-5, "expected 3.0, got {}", buf[0]);
        assert!((buf[1] - 3.0).abs() < 1e-5, "expected 3.0, got {}", buf[1]);
    }
}
