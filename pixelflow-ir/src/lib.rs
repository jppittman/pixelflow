//! # PixelFlow IR
//!
//! The shared Intermediate Representation (IR).
//!
//! - **Traits**: `Op` trait defines behavior, `EmitStyle` for codegen.
//! - **Ops**: Unit structs (`Add`, `Mul`) implement `Op`.
//!
//! A SIMD backend abstraction (`Backend`/`SimdOps`) lived here, then moved to
//! `pixelflow-core` on 2026-08-02 (it was not IR and not codegen — it lived
//! beside `Field`, which it backed). It backed a per-batch "combinator"
//! evaluation tier that the JIT superseded; both the tier and the abstraction
//! are gone (docs/plans/2026-09-06-kernel-with-a-lattice.md). `Field` now
//! reaches its two remaining constructors as inherent methods on
//! `pixelflow-core`'s own `pub(crate)` lane types, with no trait at all.

// NOTE: `no_std` support (disabling the `std` feature) is currently
// incomplete: `cargo check -p pixelflow-ir --no-default-features` fails with
// over 200 errors (missing f32 methods, `ExprId` deref mismatches in
// `arena.rs`). No CI job builds this crate with `std` off -- the default
// feature set and `--all-features` both enable it -- so this has never been
// exercised. Tracked as a known, non-blocking gap by the
// `pixelflow-ir-nostd-status` job in `.github/workflows/rust.yaml`; treat
// `no_std` as aspirational until that job is green.
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

/// Exact dyadic rationals — the constant domain the e-graph folds in, so
/// that folding cannot contradict the algebraic rewrites. See the module docs.
pub mod dyadic;
pub mod kind;
pub mod traits;
pub mod variance;

pub use variance::{LatticeShape, Variance, compute_dag_variance};

pub mod arena;

/// A generic arena-backed DAG whose consumers never see the arena: nodes
/// are named by a borrowed [`dag::Node`] handle, never a raw index.
///
/// `ExprArena` predates this and still keeps its own index-based storage.
/// What holds it back is representation coupling, not lifecycle — its
/// mutating methods take `&mut self` but are already pure at every call
/// site, so `Builder` would serve them. See
/// `docs/plans/2026-09-09-exprarena-on-dag.md` for the actual blockers and
/// the staging around them.
///
/// A *new* graph should reach for this rather than hand-roll another `Vec`
/// plus index-newtype.
pub mod dag;
pub use dag::{Builder, Dag, Key, Node, Rooted, Scratch, SideTable};

pub mod expr;
pub use expr::{
    Environment, ExprBuilderExt, ExprData, compute_dag_depth, copy_subgraph, copy_subgraph_in,
    depth, from_arena, has_degenerate, has_var, node_count_subtree, retired_axis, splice,
    substitute_params, substitute_vars, subtree_eq, to_arena,
};

/// IR-to-IR transforms: each takes an expression graph and returns another.
/// Target-blind by construction — nothing here knows which ISA it is feeding.
pub mod passes;
pub use arena::{ExprArena, ExprId, ExprNode};

/// The term language the e-graph speaks: destructure a node, rebuild a node.
/// Naming it is what makes an optimizer expressible as an endomorphism on the
/// IR rather than as a hand-rolled conversion per tier.
pub mod term;
mod term_arena;
pub use term::{Children, Ir, Shape};

/// Optimization as an endomorphism on the IR — including the identity, which
/// is what `kernel_raw!` means and what a measurement's control arm needs.
pub mod optimize;
pub use optimize::{Identity, Optimize, Rewritten, Then};

pub mod binding;
pub use binding::{BindError, BindingTable};

// The differential-testing oracle, not an execution tier: PixelFlow is
// JIT-only, so nothing in a shipped build may reach a tree-walking evaluator.
// Gating it here is what enforces that — `cargo build` cannot name it.
#[cfg(any(test, feature = "oracle"))]
pub mod eval;
#[cfg(any(test, feature = "oracle"))]
pub use eval::{
    DifferentialCheck, MaskComparison, MaskVerdict, PointCheck, PointVerdict, Tolerance,
    compare_mask_root, equivalence_tolerance, eval_scalar, is_mask_valued, is_valid_mask,
    op_is_divergent_at, trunc_input_is_divergent,
};

pub mod kernel;
pub use kernel::{Bits, Kernel, Monoid, Scalar, Uniform};

pub use kind::OpKind;
pub use kind::known_method_names;
pub use traits::EmitStyle;
