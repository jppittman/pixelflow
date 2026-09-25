---
trigger: always_on
---

# PixelFlow Core

**Pure Algebra for SIMD Graphics**

Write math. Get SIMD. No compromise.

```rust
use pixelflow_core::{Manifold, X, Y};

// Circle signed distance field
let circle = (X * X + Y * Y).sqrt() - 100.0;

// Evaluate at any coordinate
let distance = circle.eval_raw(x, y, 0.0, 0.0);
```

## Philosophy

Inspired by [Conal Elliott's denotational design](http://conal.net/papers/icfp97/) and [Halide](https://halide-lang.org/), `pixelflow-core` treats SIMD not as an optimization but as the **algebraic realization of continuous fields**.

You write equations. The type system builds a compute graph. Evaluation compiles to optimal vector assembly.

## Core Concepts

### Field

The computational atom—a SIMD vector of `f32` values. All operations work lane-wise.

```rust
// Fields are created implicitly when you evaluate manifolds
let result: Field = manifold.eval_raw(x, y, z, w);
```

### Manifold

The one trait. Everything is a function over 4D coordinates:

```rust
pub trait Manifold: Send + Sync {
    fn eval(&self, x: Field, y: Field, z: Field, w: Field) -> Field;
}
```

### Variables

Built-in coordinate accessors:

```rust
use pixelflow_core::{X, Y, Z, W};

// X just returns the x coordinate
// Y just returns the y coordinate  
// etc.
```

### Operator Overloads

Write math naturally. Operators build an AST that evaluates to SIMD:

```rust
// All these "just work"
let sum = X + Y;
let product = X * 2.0;
let complex = (X + 2.0) * Y - 0.5;
let ratio = X / (Y + 1.0);
```

## Examples

### Circle SDF

```rust
use pixelflow_core::{Manifold, ManifoldExt, X, Y};

// Signed distance to a circle at origin, radius 100
let circle = (X * X + Y * Y).sqrt() - 100.0;

// Inside: negative. Outside: positive.
```

### Centered Circle

```rust
use pixelflow_core::{Manifold, ManifoldExt, X, Y};

// Circle centered at (50, 50)
let cx = 50.0;
let cy = 50.0;
let radius = 30.0;

let dx = X - cx;
let dy = Y - cy;
let circle = (dx * dx + dy * dy).sqrt() - radius;
```

### Smooth Gradient

```rust
use pixelflow_core::{Manifold, ManifoldExt, X, Y};

// Horizontal gradient from 0 to 1 across 100 pixels
let gradient = X / 100.0;

// Clamp to [0, 1]
let clamped = gradient.max(0.0).min(1.0);
```

### Checkerboard

```rust
use pixelflow_core::{Manifold, ManifoldExt, X, Y};

// 10x10 pixel checkerboard
let cell_size = 10.0;

// XOR of x and y cell parity
let x_cell = (X / cell_size).floor();
let y_cell = (Y / cell_size).floor();

// Use comparison and select for the pattern
let x_odd = (x_cell % 2.0).ge(1.0);
let y_odd = (y_cell % 2.0).ge(1.0);

// XOR via: (x && !y) || (!x && y)
// Or simpler: different parity = white
let checker = x_odd.select(
    y_odd.select(0.0, 1.0),  // x odd: white if y even
    y_odd.select(1.0, 0.0),  // x even: white if y odd
);
```

### Mandelbrot (using Fix)

```rust
use pixelflow_core::{Manifold, ManifoldExt, Fix, X, Y, W};

// W is the iteration state (escape radius squared)
// Simplified: just check if |z|² > 4 after N iterations

let mandelbrot = Fix {
    seed: 0.0,                        // z starts at 0
    step: W * W + X,                  // z = z² + c (simplified to 1D)
    done: W.abs().gt(2.0),            // escape when |z| > 2
};

// Evaluate: returns the final z value (or escape value)
```

## Ergonomics

### Seamless Scalar Promotion

Scalars auto-promote to `Field`:

```rust
// No splat() needed - 2.0 and 0.5 just work
let expr = X * 2.0 + 0.5;
```

### Select with Scalars

The `select` method accepts scalars directly:

```rust
let mask = X.lt(100.0);
let result = mask.select(1.0, 0.0);  // No Field::splat needed
```

### ManifoldExt Methods

Chaining methods for fluent APIs:

```rust
use pixelflow_core::ManifoldExt;

let result = X
    .add(10.0)        // X + 10
    .mul(Y)           // (X + 10) * Y
    .sqrt()           // sqrt((X + 10) * Y)
    .max(0.0)         // clamp to non-negative
    .min(1.0);        // clamp to max 1
```

### Materializing to Pixels

Render a manifold to a byte buffer:

```rust
use pixelflow_core::materialize;

let mut buffer = [0u8; WIDTH];
materialize(&manifold, x_start, y, &mut buffer);
// buffer now contains u8 values (0-255) from the manifold
```

## Architecture

Under the hood:
- `Field` = `SimdVec<f32>` (4 lanes on ARM NEON, 8 on AVX2)
- Expressions build an AST of `Add`, `Mul`, `Sqrt`, `If`, etc.
- Evaluation inlines to tight SIMD loops
- Zero runtime dispatch—the compiler monomorphizes everything

## License

MIT
