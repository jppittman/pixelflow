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

pub mod expr;
#[cfg(feature = "alloc")]
pub use expr::Expr;

pub mod jit_manifold;
pub use jit_manifold::{JitManifold, ScanlineJitManifold};

pub use kind::OpKind;
pub use ops::{ALL_OPS, OP_COUNT, known_method_names, op_by_index, op_by_name};
pub use traits::{EmitStyle, Op, OpMeta};
