//! # PixelFlow IR
//!
//! The shared Intermediate Representation (IR) and backend abstraction.
//!
//! - **Traits**: `Op` trait defines behavior, `EmitStyle` for codegen.
//! - **Ops**: Unit structs (`Add`, `Mul`) implement `Op`.
//! - **ALL_OPS**: The single source of truth for all operations.
//! - **Backend**: SIMD execution traits.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod backend;
pub mod kind;
pub mod ops;
pub mod traits;
pub mod variance;

pub use variance::Variance;
pub use variance::{compute_arena_variance, find_hoistable_arena_nodes};

pub mod arena;
pub use arena::{ExprArena, ExprId, ExprNode, HasIr};

pub mod binding;
pub use binding::{BindError, BindingTable};

pub mod eval;
pub use eval::eval_scalar;

pub mod jit_manifold;
pub use jit_manifold::{JitManifold, ScanlineJitManifold};

#[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub mod jit_cache;

pub use kind::OpKind;
pub use ops::{ALL_OPS, OP_COUNT, known_method_names, op_by_index, op_by_name};
pub use traits::{EmitStyle, Op, OpMeta};

/// Byte width of the SIMD vector this build's JIT emits and calls — i.e. the
/// size of one [`KernelFn`](backend::emit::executable::KernelFn) argument and
/// return value.
///
/// The JIT has no dependency on `pixelflow-core`, so it cannot name `Field`
/// directly. This const is the single source of truth for the width the emitter
/// and the `KernelFn` ABI agree on. Callers that bridge `Field` to a JIT kernel
/// (the `kernel_jit!` wrapper) assert `size_of::<Field>() == JIT_VECTOR_BYTES`
/// at compile time, turning any width disagreement into a clear build error
/// rather than a raw `transmute` size error (or, worse, a silent miscompile).
///
/// 64 (512-bit, AVX-512) when compiled with `target_feature = "avx512f"`, where
/// `compile_arena_dag` routes to `Avx512Backend` and `KernelFn` is `__m512`;
/// otherwise 16 (128-bit, SSE2/NEON). This matches `pixelflow-core`'s `Field`
/// width under the same build flags, so the `kernel_jit!` assert holds.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub const JIT_VECTOR_BYTES: usize = 64;
/// See the AVX-512 variant above.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub const JIT_VECTOR_BYTES: usize = 16;
