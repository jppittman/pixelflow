# pixelflow-core

**Is.** The evaluation boundary (DENOTATION §5): the lattice, the compiled
manifold and the one verb between a kernel and a buffer. A lattice is an extent
and nothing else; `collapse` is tabulation, and the `DiscreteManifold` it
returns is the function, `index(collapse(f)) = f`. It defines no language
(`Kernel` is `pixelflow-ir`'s, re-exported) and holds no vector, width, colour
or platform code.

## Owns
- The domain: `Lattice`, an extent `[x, y]` with no origin or frame, and
  `IndexRange`, its one sub-lattice (indices, never coordinates).
- The compiled object: `Manifold`, a `Kernel` compiled at a lattice's shape
  behind the shape-keyed cache, and `UniformBlock`, its scalar arguments.
- The verb: `Lattice::collapse`, one call per stripe into the caller's plane;
  `bake` is its sugar for a kernel that binds nothing.
- `DiscreteManifold`, `BilinearSampler` (the named lift from `R²`) and
  `FastMathGuard` (scoped FTZ/DAZ).

## Never
- No `eval`: only `collapse` (and the 1x1 collapse behind `Lattice::eval_at`)
  turns a manifold into numbers. No per-batch or per-point entry, no
  interpreter.
- No lane, vector type or width; a test needing the lane count reads
  `pixelflow_codegen::jit_vector_bytes()`.
- No colour (`pixelflow-graphics`), no window or OS code (`pixelflow-runtime`),
  no dependency on `pixelflow-compiler`.
- No coordinate on the index: an origin, frame or extra axis is a contramap on
  the kernel (`Kernel::at`) or a `Uniform`.
- No allocation in a bind or a band collapse; a degenerate extent, a declared
  buffer with no data, or too many buffers panics naming the reason.

## Contract
`src/lib.rs` re-exports `Lattice`, `DiscreteManifold`, `IndexRange`,
`BilinearSampler`, `Manifold`, `UniformBlock`, `PlaneRegion`, `BoundManifold`,
`MAX_BOUND_BUFFERS`, `ArityMismatch`, `UnknownUniform`, `CellGridProgram` and
its `CellGrid*` companions, `FastMathGuard`, and `Kernel`, `Uniform`, `Bits`,
`Monoid` from `pixelflow-ir`. The module is `lattice`; `__macro` is doc-hidden
plumbing so a `kernel!` expansion resolves from the two-crate surface. A new
language op goes in `pixelflow-ir`, not here.

## Enforced by
- Types: `Lattice` has one field, `extent`, so no origin is representable;
  `Manifold` has no `eval`; `compile_error!` in `src/lib.rs` refuses any target
  but x86-64 and aarch64.
- `index_of_collapse_is_the_kernel_everywhere` (`src/lattice/tests.rs`) pins
  the law; `tests/kernel_bake.rs` and `tests/kernel_by_reference.rs` pin that a
  baked or by-name kernel renders what its composition renders.
- `tests/bind_allocates_nothing.rs` counts allocations across bind and collapse;
  desloppify `per-frame-allocation` covers `pixelflow-core/**`.
- desloppify `simd-is-codegens`, `no-terminal-logic-in-pixelflow`.

## Behind
- `BoundManifold`, `Manifold::bind` and `MAX_BOUND_BUFFERS` are public and
  `collapse` takes the bound form, though a bound buffer is not a concept
  (DENOTATION §6, §11); closed by kernels-evaluate-kernels §2 (JP-22).
- `CellGridProgram` is terminal-shaped geometry in a crate that may hold none;
  it leaves core (the-language-is-kernel D13).
- `#![no_std]` is not whole: `src/lattice/manifold.rs` takes `extern crate std`
  for its `RwLock` and `HashMap`; the crate does not build for wasm32
  (`scripts/wasm_status_baseline.txt`).
- `Lattice::extent` is `[u32; 2]`, against the 64-bit control plane; no plan
  or Linear issue closes it.
- The traversal reaches the kernel as IR folds; it moves into the CPU backend
  (kernels-evaluate-kernels §3, JP-16).

## Decided in
`docs/DENOTATION.md` §5, §6, §11; `docs/archive/plans/2026-09-06-kernel-with-a-lattice.md`,
`docs/archive/plans/2026-09-06-lattice-is-the-index.md`,
`docs/archive/plans/2026-09-16-collapse-is-a-fold.md`,
`docs/language/2026-10-15-kernels-evaluate-kernels.md` §2–§3.
