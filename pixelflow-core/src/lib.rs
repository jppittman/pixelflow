//! # PixelFlow Core: kernels, lattices, and the one verb between them
//!
//! Three objects and one verb, and nothing else is an evaluation API:
//!
//! ```text
//! Kernel ──Manifold::compile(extent)──▶ Manifold ──bind(&[(id, buf)])──▶ BoundManifold
//!        ──Lattice::collapse──▶ DiscreteManifold
//! ```
//!
//! - A **[`Kernel`]** is the description: an arena with a root, built with the
//!   language's vocabulary — `Kernel::x()`, arithmetic, `.at`, `.select`,
//!   `.sqrt`, `over`, `dx`/`dy` — or by the `kernel!` macro, which is the same
//!   value with closure syntax. It carries no code and no shape.
//! - A **[`Manifold`]** is a kernel *compiled at a lattice's shape*: the thing
//!   that can be sampled over a domain, specialized on the extents it was
//!   compiled for, held behind the global compile cache. It has no `eval` and
//!   is not batch-shaped.
//! - A **[`Lattice`]** is the domain: an extent, and nothing else. The shape
//!   is data, not a type — a frame and an index range are one `Lattice` with
//!   different extents. A coordinate frame is never part of it; where one is
//!   needed it is a contramap on the kernel instead.
//! - **[`Lattice::collapse`]** is the verb: tabulate a bound manifold over a
//!   lattice into a [`DiscreteManifold`], the buffer that *is* a manifold by
//!   the representable-functor law `index(collapse(f)) = f`.
//!   [`Lattice::bake`] is that line for a kernel that reads no memory.
//!
//! The loop nest, the invariant hoisting, the pack and the register
//! allocation all live *inside* the emitted code, so a collapse is one call
//! per plane rather than one call per row or per SIMD batch. There is no
//! per-batch entry, and there is no interpreter: a kernel becomes numbers by
//! being compiled at a shape and collapsed, or not at all.
//!
//! ## What is not here
//!
//! **No colour.** This crate knows fields, lattices and integer/bit ops — not
//! RGBA, byte lanes or pixel formats. A colour output is four channel kernels
//! in `[0, 1]` packed by integer IR ops that `pixelflow-graphics` composes.
//!
//! **No SIMD at all.** This crate holds no vector type and no width: the
//! batch a collapse executes by is the JIT's, decided at startup by the CPU
//! (`pixelflow_codegen::isa`), and a buffer here is `f32`s at a pitch.
//! Nothing public names a lane or a vector, and nothing private does either.
//!
//! **No expression templates.** Manifolds were once zero-sized types that
//! monomorphized into a fused kernel and evaluated one SIMD batch at a time.
//! That tier is gone (docs/plans/2026-09-06-kernel-with-a-lattice.md): the
//! compiler already beat it, and the per-batch call boundary was most of why.
//!
//! ## Key modules
//!
//! - **[`lattice`]** — the lattice, the compiled manifold, the collapsed
//!   buffer, and the cell grid the terminal renders through
//!
//! ## Execution notes
//!
//! - **Targets**: x86-64 and aarch64 only. Rendering goes through the JIT,
//!   which has no other backends and no interpreter fallback.
//! - **Zero per-frame allocation**: a bound manifold binds buffers by
//!   identity and a band collapse allocates nothing.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

// Tests use std (println, env, fs) for harnesses; shipped code stays no_std.
#[cfg(test)]
extern crate std;

// ============================================================================
// Modules
// ============================================================================

/// Flush-to-zero / denormals-are-zero, as a scoped guard on the FP control
/// register.
pub mod fastmath;

/// Lattice: representable functor for kernel evaluation.
pub mod lattice;

// ============================================================================
// Re-exports (The "Prelude")
// ============================================================================

pub use fastmath::FastMathGuard;
pub use pixelflow_ir::{Bits, Kernel, Monoid, Scalar, Uniform};

// Lattice types: the compiled object, what binding it produces, the domain,
// and the buffer a collapse fills.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub use lattice::BilinearSampler;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub use lattice::cell_grid::{
    CELL_STRIDE, CellGridBuffers, CellGridFrame, CellGridKernels, CellGridMetrics, CellGridParams,
    CellGridProgram, CellGridShape, CellGridSlots,
};
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub use lattice::manifold::{
    BoundManifold, MAX_BOUND_BUFFERS, Manifold, PlaneRegion, UniformBlock, UnknownUniform,
};
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub use lattice::union::IndexRange;
pub use lattice::{DiscreteManifold, Lattice};

// Macro plumbing, serde-`__private`-style: a `kernel!` expansion runs in the
// *consumer's* crate, whose extern prelude is only guaranteed to contain the
// documented two-crate surface (pixelflow-core + pixelflow-compiler).
// Generated code therefore reaches pixelflow-ir through this module — a bare
// `::pixelflow_ir` path would fail to resolve for any consumer that does not
// also declare that crate as a direct dependency. Nothing reaches
// pixelflow-codegen: an expansion builds an arena, and only a compile at a
// lattice's shape emits code.
//
// This does NOT relax the "pixelflow-core shouldn't re-export the IR" ruling:
// that ruling is about the public API surface, and this module exists solely
// so macro expansions resolve from the two-crate dependency surface. It is
// not public API — do not use it directly.
#[doc(hidden)]
pub mod __macro {
    pub use pixelflow_ir as ir;
}

// No scalar fallback: the JIT (`Manifold::compile`, and therefore
// `Lattice::bake`) exists only on x86-64 and aarch64, and there is no
// interpreter render path. A target without a JIT could hold a lattice but
// could not render anything, so it fails here, loudly, rather than silently
// building something inert. Which x86-64 tier a process runs — AVX2+FMA, or
// AVX-512 where the host has it — is the JIT's decision, made once at
// startup from CPUID (`pixelflow_codegen::isa`); this crate carries no
// vector type to agree or disagree with it.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!(
    "pixelflow-core supports x86-64 and aarch64 only: rendering goes through \
     the JIT, which has no other targets and no interpreter fallback"
);
