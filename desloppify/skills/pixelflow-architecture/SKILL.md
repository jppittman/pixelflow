---
name: pixelflow-architecture
description: Crate boundaries and design constraints of the pixelflow workspace (CLAUDE.md "Critical Constraints").
---

# PixelFlow architecture

- **Crates.** `pixelflow-*` is a general-purpose graphics library being extracted
  to its own repository: no terminal logic (PTY, ANSI, terminal grid/state) in
  any `pixelflow-*` crate; that belongs in `core-term`. Platform code lives in
  `pixelflow-runtime`. `pixelflow-core` holds lattices, the compiled `Manifold`
  and `collapse`; it is `no_std` + `alloc` and has no vector type.
- **The language.** Users compose `Kernel` values (or write `kernel!`); the
  compiler owns the loop nest, packing and register allocation. A kernel is a
  description, `Manifold::compile` specializes it at a lattice's shape,
  `Lattice::collapse` is the only verb that produces numbers. The language is a
  DAG: no iteration binder.
- **SIMD is an implementation detail.** Nothing outside `pixelflow-codegen`'s
  emitters names a lane, a vector width, or an intrinsic. The width is the JIT's
  (`pixelflow_codegen::jit_vector_bytes()`), chosen once at startup by CPUID —
  never by build flags. No raw lane arithmetic: build the arena and let the
  compiler emit instructions.
- **Minimal public API.** Don't widen visibility of internals; compose `Kernel`
  values instead of exposing fields.
- **Denote before you build.** Say what a thing means, in the type system,
  before writing code that manipulates it. A convention written in a comment
  ("must be a literal", "these bits are a mask", "in the caller's space") is an
  invariant something will eventually break. When a type's meaning is extended,
  extend the type.
- **The control plane is 64-bit.** Every index, id, count, extent and bound
  describing a program (`ExprId`, `ValueId`, fold ends, binder slots, class ids)
  is 64 bits. Narrower needs a documented, profiler-measured reason in the
  type's own doc. The data plane is narrow on purpose: `f32` lanes, `u32`
  pixels, ISA encoding fields (`imm12`, `rel32`, register numbers).
- **Masks are bit patterns, not numbers.** A comparison yields all-ones/all-zero
  lanes; `OpKind::mask(bool)` is the only constructor. Spelling a true mask
  `1.0` corrupts bitwise blends.
- **Take what the hardware gives.** Don't spend hot-path instructions matching
  scalar IEEE edge cases, and never hand-roll a worse form of an instruction
  that exists (`(x + 0.5).floor()` for `round`). Precision is tunable; range is
  not — out of domain returns NaN, not a clamped value.
- **Zero allocations per frame.** Rendering reuses buffers (ping-pong frames).
- **Actor lanes.** Control > Management > Data. Control and Management are for
  latency (input, resize, lifecycle, config); Data is the bulk stream and the
  only lane with backpressure.
