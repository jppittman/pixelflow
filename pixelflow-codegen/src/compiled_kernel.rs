//! CompiledKernel: one kernel's emitted code, held as executable memory.
//!
//! This is the *code*, not the object a consumer names: `pixelflow-core`'s
//! `Manifold` is a kernel compiled at a lattice shape, and this is what it
//! holds inside. It owns an [`ExecutableCode`] and exposes it through
//! [`call`](CompiledKernel::call) — the whole collapse, one call.
//!
//! There are no row/grid/point evaluators here and no per-batch entry: the
//! loop nest is inside the code, because the lattice's rows, batches and
//! lanes are folds the kernel was wrapped in before it was scheduled
//! (docs/plans/2026-09-16-collapse-is-a-fold.md).

use crate::emit::executable::ExecutableCode;
use pixelflow_ir::LatticeShape;

/// One kernel's emitted code, at one lattice shape. Owns the executable
/// memory; no cache — the caller decides its lifetime.
pub struct CompiledKernel {
    code: ExecutableCode,
    shape: LatticeShape,
}

impl CompiledKernel {
    /// Wrap newly compiled executable code into a `CompiledKernel` for a
    /// lattice of `shape`.
    ///
    /// The shape is what the code *is*: its loop bounds are the extent, so a
    /// call fills exactly `shape` samples and nothing about that is decided
    /// at the call.
    #[must_use]
    pub const fn new(code: ExecutableCode, shape: LatticeShape) -> Self {
        Self { code, shape }
    }

    /// The lattice this kernel was compiled for.
    #[must_use]
    pub const fn shape(&self) -> LatticeShape {
        self.shape
    }

    /// The emitted machine code, for offline inspection (disassembly,
    /// profiler correlation). The bytes are the artifact, not an ABI.
    #[must_use]
    pub fn code_bytes(&self) -> &[u8] {
        self.code.as_bytes()
    }

    /// Collapse: fill every sample of the shape this kernel was compiled at,
    /// into `out`, whose rows are `pitch` elements apart.
    ///
    /// # Safety
    ///
    /// - `ctx` must hold valid base pointers for every buffer declared by the
    ///   arena, in slot order, followed by the uniform block's base pointer
    ///   (read only when the arena declares a uniform) and then the origin
    ///   block's — two `f32`s, `x0` then `y0`.
    /// - `out` must be writable for `(height - 1) * pitch + width` elements,
    ///   `[width, height]` being [`shape`](Self::shape)'s extent.
    #[inline(always)]
    pub unsafe fn call(&self, ctx: *const *const f32, out: *mut f32, pitch: usize) {
        // SAFETY: delegated to `ExecutableCode`, which invokes the emitted
        // `KernelFn` under exactly this contract.
        unsafe { self.code.call(ctx, out, pitch) }
    }
}

// SAFETY: ExecutableCode is read-only mapped memory with no interior mutability.
unsafe impl Send for CompiledKernel {}
unsafe impl Sync for CompiledKernel {}
