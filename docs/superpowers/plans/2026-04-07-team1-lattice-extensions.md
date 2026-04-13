# Team 1: Lattice Extensions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `IndexLattice1D`, `IndexLattice2D`, and `collapse_axis` to `pixelflow-core` so ML tensor operations (matmul, affine layers) can be expressed as lattice collapses.

**Architecture:** Follows the existing `FrameLattice`/`ScanlineLattice` pattern exactly. `IndexLattice1D(n)` loops X over `[0, n)` with Y/Z/W fixed at 0 — semantically "feature indices", not pixel coordinates. `IndexLattice2D(m, n)` loops X over `[0, m)` and Y over `[0, n)`. `collapse_axis(axis, op, manifold)` on `IndexLattice2D` reduces one dimension and returns a `DiscreteManifold` indexed by the surviving dimension (always stored with the result in X, height=1).

**Tech Stack:** Rust stable, `pixelflow-core`, no new dependencies.

---

## Context You Need

Read these before starting:
- `pixelflow-core/src/lattice.rs` — full file. Your additions go here.
- `pixelflow-core/src/lib.rs` — add your new types to the `pub use lattice::` re-export.

Key types already in `lattice.rs`:
- `LatticeDomain` trait: `len()`, `loop_vars()`, `coord(index) -> (f32,f32,f32,f32)`
- `Lattice` trait: `collapse(manifold) -> DiscreteManifold`, `collapse_with(op, manifold) -> Field`
- `ReduceOp` enum: `Add`, `Mul`, `Min`, `Max` with `.identity() -> Field` and `.apply(acc, val) -> Field`
- `Field::from(f32)`, `Field::sequential(0.0)`, `field.raw_add(step)`, `field.store(&mut [f32; PARALLELISM])`
- `Field::select_raw(mask, a, b)`, `field.lt(threshold)`
- `PARALLELISM` constant (SIMD lane width, e.g. 4 or 8)
- `DiscreteManifold { buffer: Vec<f32>, width: usize, height: usize }` — construct directly (same crate)
- `Manifold<(Field, Field, Field, Field), Output = Field>` — the trait bound for manifold args

**Convention for collapse_axis result:**
The surviving dimension is always remapped to X in the output `DiscreteManifold`:
- `collapse_axis(0, op, m)` → reduce X, keep Y → output `{ width: self.height, height: 1 }`, `result[j]` stored at index `(j, 0)`
- `collapse_axis(1, op, m)` → reduce Y, keep X → output `{ width: self.width, height: 1 }`, `result[i]` stored at index `(i, 0)`

---

## File Structure

| File | Change |
|------|--------|
| `pixelflow-core/src/lattice.rs` | Add `IndexLattice1D`, `IndexLattice2D`, `collapse_axis` impl on `IndexLattice2D`, and tests |
| `pixelflow-core/src/lib.rs` | Add `IndexLattice1D`, `IndexLattice2D` to `pub use lattice::` |

---

## Task 1: IndexLattice1D

**Files:** Modify `pixelflow-core/src/lattice.rs`

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)]` block at the bottom of `lattice.rs`:

```rust
#[test]
fn index_lattice1d_collapse_identity() {
    use crate::variables::X;
    // IndexLattice1D(4).collapse(X) should produce [0.0, 1.0, 2.0, 3.0]
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
    use crate::variables::X;
    // sum of [0,1,2,3] = 6. collapse_with(Add, X) should fold to a single Field of 6.
    let lattice = IndexLattice1D::new(4);
    let result = lattice.collapse_with(ReduceOp::Add, &X);
    let mut packed = [0.0f32; PARALLELISM];
    result.store(&mut packed);
    assert!((packed[0] - 6.0).abs() < 1e-5, "expected 6.0, got {}", packed[0]);
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test -p pixelflow-core index_lattice1d
```

Expected: `error[E0412]: cannot find type IndexLattice1D`

- [ ] **Step 3: Implement IndexLattice1D**

Add after the `PointLattice` block in `lattice.rs`:

```rust
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

/// Stored loop variable indices for IndexLattice1D.
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

        acc
    }
}
```

- [ ] **Step 4: Export IndexLattice1D from lib.rs**

In `pixelflow-core/src/lib.rs`, find the `pub use lattice::` line and add `IndexLattice1D`:

```rust
pub use lattice::{
    DiscreteManifold, FrameLattice, IndexLattice1D, Lattice, LatticeDomain,
    PointLattice, ReduceOp, ScanlineLattice,
};
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cargo test -p pixelflow-core index_lattice1d
```

Expected: `test index_lattice1d_collapse_identity ... ok` and `test index_lattice1d_collapse_with_sum ... ok`

- [ ] **Step 6: Commit**

```bash
git add pixelflow-core/src/lattice.rs pixelflow-core/src/lib.rs
git commit -m "feat(core): IndexLattice1D — 1D index lattice for tensor ops"
```

---

## Task 2: IndexLattice2D

**Files:** Modify `pixelflow-core/src/lattice.rs`

- [ ] **Step 1: Write the failing tests**

Add to the `#[cfg(test)]` block:

```rust
#[test]
fn index_lattice2d_collapse_xy_sum() {
    use crate::variables::{X, Y};
    use crate::ext::ManifoldExt;
    // 3×2 lattice (width=3, height=2). Values = X + Y.
    // Row 0 (Y=0): [0,1,2]. Row 1 (Y=1): [1,2,3].
    let lattice = IndexLattice2D::new(3, 2);
    let result = lattice.collapse(&(X + Y));
    assert_eq!(result.width(), 3);
    assert_eq!(result.height(), 2);
    let buf = result.buffer();
    // Row-major: [y=0,x=0], [y=0,x=1], [y=0,x=2], [y=1,x=0], [y=1,x=1], [y=1,x=2]
    assert!((buf[0] - 0.0).abs() < 1e-6); // x=0, y=0
    assert!((buf[1] - 1.0).abs() < 1e-6); // x=1, y=0
    assert!((buf[2] - 2.0).abs() < 1e-6); // x=2, y=0
    assert!((buf[3] - 1.0).abs() < 1e-6); // x=0, y=1
    assert!((buf[4] - 2.0).abs() < 1e-6); // x=1, y=1
    assert!((buf[5] - 3.0).abs() < 1e-6); // x=2, y=1
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test -p pixelflow-core index_lattice2d
```

Expected: `error[E0412]: cannot find type IndexLattice2D`

- [ ] **Step 3: Implement IndexLattice2D**

Add after `IndexLattice1D` in `lattice.rs`:

```rust
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

/// Stored loop variable indices for IndexLattice2D.
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

        // Y outer, X inner
        for y in 0..self.height {
            let y_field = Field::from(y as f32);
            let row_offset = y * self.width;
            let mut x = 0usize;
            let mut x_field = Field::sequential(0.0);

            while x + PARALLELISM <= self.width {
                let result = manifold.eval((x_field, y_field, z_field, w_field));
                result.store(&mut packed);
                buffer[row_offset + x..row_offset + x + PARALLELISM]
                    .copy_from_slice(&packed);
                x += PARALLELISM;
                x_field = x_field.raw_add(step);
            }

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

        // → horizontal reduce: store lanes, sum with match op, return Field::from(scalar)
        acc
    }
}
```

- [ ] **Step 4: Export IndexLattice2D from lib.rs**

```rust
pub use lattice::{
    DiscreteManifold, FrameLattice, IndexLattice1D, IndexLattice2D, Lattice,
    LatticeDomain, PointLattice, ReduceOp, ScanlineLattice,
};
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p pixelflow-core index_lattice2d
```

Expected: `test index_lattice2d_collapse_xy_sum ... ok`

- [ ] **Step 6: Commit**

```bash
git add pixelflow-core/src/lattice.rs pixelflow-core/src/lib.rs
git commit -m "feat(core): IndexLattice2D — 2D index lattice for weight matrices"
```

---

## Task 3: collapse_axis (the matmul primitive)

**Files:** Modify `pixelflow-core/src/lattice.rs`

- [ ] **Step 1: Write the failing test — dot product**

Add to `#[cfg(test)]`:

```rust
#[test]
fn collapse_axis0_dot_product() {
    // W = [[1,2],[3,4]] (width=2 input, height=2 output)
    // x = [1, 2] (width=2)
    // W^T @ x: output[0] = 1*1 + 3*2 = 7, output[1] = 2*1 + 4*2 = 10
    //
    // W indexed as W(input_i=X, output_j=Y):
    // W(0,0)=1, W(1,0)=3, W(0,1)=2, W(1,1)=4
    // x(X): x(0)=1, x(1)=2
    //
    // collapse_axis(0, Add, W*x) for each Y=j: sum over X=i of W(i,j)*x(i)
    let w_buf = vec![1.0f32, 3.0, 2.0, 4.0]; // row-major: [y=0,x=0],[y=0,x=1],[y=1,x=0],[y=1,x=1]
    let w = DiscreteManifold::new(w_buf, 2, 2);
    let x_buf = vec![1.0f32, 2.0];
    let x = DiscreteManifold::new(x_buf, 2, 1);

    let lattice = IndexLattice2D::new(2, 2); // width=INPUT=2, height=OUTPUT=2

    struct Product {
        w: DiscreteManifold,
        x: DiscreteManifold,
    }
    impl Manifold<(Field, Field, Field, Field)> for Product {
        type Output = Field;
        fn eval(&self, (xi, yj, _, _): (Field, Field, Field, Field)) -> Field {
            let zero = Field::from(0.0);
            self.w.eval((xi, yj, zero, zero))
                * self.x.eval((xi, zero, zero, zero))
        }
    }

    let result = lattice.collapse_axis(0, ReduceOp::Add, &Product { w, x });
    // result: DiscreteManifold { width: 2, height: 1 }, indexed by X=j
    assert_eq!(result.width(), 2);
    assert_eq!(result.height(), 1);
    let buf = result.buffer();
    assert!((buf[0] - 7.0).abs() < 1e-4, "expected 7.0, got {}", buf[0]);
    assert!((buf[1] - 10.0).abs() < 1e-4, "expected 10.0, got {}", buf[1]);
}

#[test]
fn collapse_axis1_row_sum() {
    // 2×3 lattice (width=2, height=3). Values = Y.
    // collapse_axis(1, Add): for each X=i, sum Y over [0,3) = 0+1+2 = 3
    // result: width=2, height=1. result[0]=3, result[1]=3.
    use crate::variables::Y;
    let lattice = IndexLattice2D::new(2, 3);
    let result = lattice.collapse_axis(1, ReduceOp::Add, &Y);
    assert_eq!(result.width(), 2);
    assert_eq!(result.height(), 1);
    let buf = result.buffer();
    assert!((buf[0] - 3.0).abs() < 1e-5, "expected 3.0, got {}", buf[0]);
    assert!((buf[1] - 3.0).abs() < 1e-5, "expected 3.0, got {}", buf[1]);
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test -p pixelflow-core collapse_axis
```

Expected: `error[E0599]: no method named collapse_axis found`

- [ ] **Step 3: Implement collapse_axis on IndexLattice2D**

Add this `impl` block directly after the `impl Lattice for IndexLattice2D` block:

```rust
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
            _ => panic!("IndexLattice2D::collapse_axis: axis {} out of range (must be 0 or 1)", axis),
        }
    }

    /// Reduce over X, produce one value per Y. Output indexed by X (height=1).
    fn collapse_axis0<M>(&self, op: ReduceOp, manifold: &M) -> DiscreteManifold
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        // For each output index j (Y), sum over i (X).
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

        DiscreteManifold {
            buffer: out_buffer,
            width: out_len,
            height: 1,
        }
    }

    /// Reduce over Y, produce one value per X. Output indexed by X (height=1).
    fn collapse_axis1<M>(&self, op: ReduceOp, manifold: &M) -> DiscreteManifold
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        // For each output index i (X), sum over j (Y).
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

            // Horizontal reduction.
            let mut packed = [0.0f32; PARALLELISM];
            acc.store(&mut packed);
            let scalar: f32 = match op {
                ReduceOp::Add => packed.iter().sum(),
                ReduceOp::Mul => packed.iter().product(),
                ReduceOp::Min => packed.iter().cloned().fold(f32::INFINITY, f32::min),
                ReduceOp::Max => packed.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            };
            out_buffer[i] = scalar;
        }

        DiscreteManifold {
            buffer: out_buffer,
            width: out_len,
            height: 1,
        }
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p pixelflow-core collapse_axis
```

Expected: `test collapse_axis0_dot_product ... ok` and `test collapse_axis1_row_sum ... ok`

- [ ] **Step 5: Full test suite**

```bash
cargo test -p pixelflow-core
```

Expected: All existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add pixelflow-core/src/lattice.rs
git commit -m "feat(core): collapse_axis — partial reduction for matmul over IndexLattice2D"
```

---

## Task 4: Cleanup and Documentation

**Files:** `pixelflow-core/src/lattice.rs`

- [ ] **Step 1: Add module-level doc clarifying semantic distinction**

At the top of `lattice.rs`, after the existing `//!` doc comment, add a new section:

```rust
//! ## Lattice Types: Pixel-Space vs Index-Space
//!
//! **Pixel-space lattices** (`FrameLattice`, `ScanlineLattice`) represent
//! 2D or 1D grids of screen coordinates. X and Y are pixel positions.
//! Use these for rendering.
//!
//! **Index-space lattices** (`IndexLattice1D`, `IndexLattice2D`) represent
//! abstract index domains for tensor operations. X and Y are feature indices,
//! not pixel coordinates. Use these for ML tensor ops (matmul, affine layers).
//!
//! `IndexLattice2D::collapse_axis` is the core matmul primitive:
//! `output[j] = Σᵢ W(i,j) * input(i)` = `lattice.collapse_axis(0, Add, &Product{W, input})`.
```

- [ ] **Step 2: Verify lattice.rs is under 600 lines**

```bash
wc -l pixelflow-core/src/lattice.rs
```

If over 600 lines, split `IndexLattice1D` + `IndexLattice2D` into a new `pixelflow-core/src/lattice/index.rs` and `use`-import in `lattice.rs`. (At this size it likely fits — check first.)

- [ ] **Step 3: Final build and test**

```bash
cargo test -p pixelflow-core
cargo build -p pixelflow-core
```

Expected: All tests pass, no warnings on new code.

- [ ] **Step 4: Commit**

```bash
git add pixelflow-core/src/lattice.rs pixelflow-core/src/lib.rs
git commit -m "docs(core): document pixel-space vs index-space lattice distinction"
```
