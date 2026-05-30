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
pub use arena::{ExprArena, ExprId, ExprNode};

pub mod jit_manifold;
pub use jit_manifold::{JitManifold, ScanlineJitManifold};

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
/// Currently 16 (128-bit, SSE2/NEON) on every target: the wide (AVX-512)
/// emitter exists (`backend::emit::avx512`) but is not yet wired into
/// `compile_arena_dag`. When it is, this becomes 64 under
/// `target_feature = "avx512f"`. Until then an AVX-512 `pixelflow-core` build
/// (512-bit `Field`) fails the caller's assert, correctly signalling "the JIT
/// does not emit this width yet."
pub const JIT_VECTOR_BYTES: usize = 16;
