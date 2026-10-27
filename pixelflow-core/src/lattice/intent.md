# pixelflow-core/src/lattice

**Is.** The representable functor of the evaluation boundary (DENOTATION §5, §6):
`Lattice` is the index (an extent, nothing else), `collapse` is tabulation, and
`DiscreteManifold` is the table that is the function, `index(collapse(f)) = f`.
`Manifold` is the kernel compiled at a lattice's shape.

## Owns
- The law: `Lattice` has one field, so no origin turns it into `f ∘ shift`.
- Where a sample is taken. `PlaneRegion`'s origin is private; each constructor
  fixes one convention: `at_index` (the raw index, for `collapse` and
  `IndexRange::bake`), `at_point` (`eval_at` only), `rows` (pixel centres).
- The shape table (a `Manifold` compiles each new band shape on first use) and
  the destination: a band is written in place into the caller's plane at its
  stride, `f32` or (`collapse_int_rows`) `u32` bit for bit.
- `IndexRange`, the one sub-lattice; `DiscreteManifold`, write-once memory
  (a shared `Arc<[f32]>`) read back as a `Kernel`.

## Never
- No coordinate on the index: `Lattice::collapse` and `IndexRange::bake` use
  `PlaneRegion::at_index` only, and `at_index` and `at_point` are `pub(crate)`.
- No numbers from a manifold except by `collapse_rows`/`collapse_int_rows`
  (`eval_at` is the 1x1 case); `Manifold` and `DiscreteManifold` have no `eval`.
- No lane, width or ISA name in the code here; the batch and tail are the JIT's.
- No unchecked plane: a bad region, stride or plane panics in
  `BoundManifold::check`, before the unsafe call.

## Contract
`mod.rs`: `Lattice` (`frame`, `index`, `index2`, `scanline`, `eval_at`, `coord`,
`collapse`, `bake`), `AXES`, `DiscreteManifold`, `BilinearSampler`. `manifold.rs`:
`Manifold`, `UniformBlock`, `PlaneRegion`, `BoundManifold`. `union.rs`: `IndexRange`;
`cell_grid.rs`: `CellGrid*`.

## Enforced by
- Types: `Lattice { extent }`; a private `PlaneRegion.origin`; `DiscreteManifold`
  has `pub(crate)` fields and no `&mut self` method.
- `index_of_collapse_is_the_kernel_everywhere` (`tests.rs`) fails if a coordinate
  returns to the index; `a_band_of_a_new_shape_compiles_once_and_is_reused`
  (`manifold.rs`); `tests/bind_allocates_nothing.rs`; desloppify `simd-is-codegens`.

## Behind
- 2026-10-15: `mod.rs` and `manifold.rs` still draw `compile ──bind(buffers)──▶
  bound ──collapse`; `Lattice::collapse` takes a `BoundManifold`. Closed by
  kernels-evaluate-kernels §2 (JP-22).
- `DiscreteManifold::kernel` and `BilinearSampler::kernel` are read at an `R²`
  coordinate, flooring and clamping in the `Gather`; §6 puts the lift in the
  caller (D10 deletes the sampler).
- `PlaneRegion::rows` bakes the pixel centre into the origin, and `eval_at`
  places its point through `at_point`, not by contramap (no row closes this).
- `IndexRange` is one type for a requested band and a derived region (D3), in
  `union.rs`, still named for the deleted `Union`.
- `cell_grid.rs` is terminal-shaped and holds colour (`default_bg: [f32; 4]`);
  it leaves core (the-language-is-kernel D13).

## Decided in
`docs/DENOTATION.md` §5, §6, §11; `docs/archive/plans/2026-09-06-lattice-is-the-index.md`,
`2026-09-06-kernel-with-a-lattice.md`, `2026-09-16-collapse-is-a-fold.md`,
`2026-10-15-kernels-evaluate-kernels.md` §2, `2026-09-25-the-language-is-kernel.md`.
