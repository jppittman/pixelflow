//! Expression graphs to machine code.
//!
//! Everything here is downstream of the language: `pixelflow-ir` defines what
//! an expression *is*, `pixelflow-search` decides what it *should be*, and this
//! crate turns the result into instructions.
//!
//! That ordering is the point. The compile entries in [`jit_cache`] run the
//! optimizer themselves, so there is no way to ask for a compiled kernel and
//! accidentally get an unoptimized one — which is what happened while this code
//! lived *below* the e-graph and had to leave the choice to its callers.
//!
//! This is also where every OS dependency lives. Executable memory needs
//! `mmap`/`mprotect`; the language does not, and now does not link `libc`.

// The moved files were written against `alloc` paths while they lived in a
// `no_std` crate. This one is unconditionally `std` — it calls `mmap` — but the
// paths are fine as they are and rewriting ~200 imports would buy nothing.
extern crate alloc;

pub mod emit;
pub mod error;

pub mod compiled_kernel;
pub use compiled_kernel::CompiledKernel;
pub use error::CompileError;

// x86-64 and aarch64 are the architectures with emitters.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod jit_cache;

/// Byte width of the SIMD vector this build's JIT emits — the batch the
/// lattice's lane fold is executed by.
///
/// The JIT has no dependency on `pixelflow-core`, so it cannot name `Field`
/// directly. This const is the single source of truth for the width the
/// emitter strip-mines a lattice's columns to (`passes::lattice::pack`'s
/// `L`). Nothing about it reaches the [`KernelFn`](emit::executable::KernelFn)
/// ABI any more — a call passes pointers and a pitch — so a disagreement
/// with `Field` no longer miscompiles anything; `pixelflow-core` still
/// asserts the two agree, because its `Field` is the width its own buffers
/// are laid out for.
///
/// A genuine 3-way split, checked against `target_feature` — the flag that
/// actually governs what the compiler may emit. `pixelflow-core` gates
/// `NativeF32Storage` on exactly the same predicate, which is what keeps the
/// two crates' widths in step.
///
/// They once did not. `pixelflow-core` used to AND in a build-script cfg set by
/// probing the *build host's* CPU, on the theory that it was a safety net
/// against a host told to target a feature it lacks. It was not: a build script
/// is per-crate and the cfg it emits does not cross a crate boundary, so this
/// crate went on emitting 512-bit code while core's `Field` quietly narrowed to
/// 256 — the disagreement that net was supposed to prevent, caused by the net
/// itself. Building `+avx512f` on a non-AVX-512 host failed on the `transmute`
/// in core's `lattice`. Host CPU is the wrong question for a target decision;
/// running wide code on a narrow host is gated by `cargo xtask isa-matrix`,
/// which builds every level and runs only what `host_has_feature` allows.
///
/// 64 (512-bit, AVX-512) when compiled with `target_feature = "avx512f"`,
/// routing to `Avx512Backend`; 32 (256-bit, AVX2) when compiled with
/// `target_feature = "avx2"` and not `"avx512f"`, routing to `Avx2Backend`;
/// otherwise 16 (128-bit, SSE2/NEON). This matches `pixelflow-core`'s
/// `Field` width under the same build flags.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub const JIT_VECTOR_BYTES: usize = 64;
/// See the AVX-512 variant above.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub const JIT_VECTOR_BYTES: usize = 32;
/// See the AVX-512 variant above.
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx512f"),
    all(target_arch = "x86_64", target_feature = "avx2")
)))]
pub const JIT_VECTOR_BYTES: usize = 16;
