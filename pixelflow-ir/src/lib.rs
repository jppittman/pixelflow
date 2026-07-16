#![allow(clippy::if_same_then_else)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(unused_imports)]
#![allow(clippy::duplicated_attributes)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(improper_ctypes_definitions)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::approx_constant)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::identity_op)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::useless_format)]
#![allow(clippy::bad_bit_mask)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::new_without_default)]
#![allow(clippy::collapsible_match)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::must_use_candidate)]
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
