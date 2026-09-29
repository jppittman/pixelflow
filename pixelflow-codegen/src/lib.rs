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
pub mod isa;

pub mod compiled_kernel;
pub use compiled_kernel::CompiledKernel;
pub use error::CompileError;
// The one vector width in the workspace: the tier's, decided at startup by
// the CPU (see `isa`). `pixelflow-core` asks for it here rather than carrying
// a vector type of its own to assert against — which is how the two crates'
// widths used to drift, each keyed on its own reading of `target_feature`.
pub use isa::jit_vector_bytes;

// x86-64 and aarch64 are the architectures with emitters.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod jit_cache;
