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

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

/// Exact dyadic rationals — the constant domain the e-graph folds in, so
/// that folding cannot contradict the algebraic rewrites. See the module docs.
pub mod dyadic;
pub mod kind;
pub mod traits;
pub mod variance;

pub use variance::{LatticeShape, Variance, compute_dag_variance};

/// The declarations an expression's leaves index, and the index spaces `Var`
/// is drawn from. Domain types, not storage.
pub mod decl;
pub use decl::{
    BufferDecl, BufferId, BufferIdentity, COORD_AXES, REDUCE_BINDER_BASE, REDUCE_BINDERS,
    RETIRED_COORD_AXES, UniformDecl, UniformId, UniformIdentity,
};

/// A generic arena-backed DAG whose consumers never see the arena: nodes
/// are named by a borrowed [`dag::Node`] handle, never a raw index.
///
/// This is the *only* expression storage. `ExprArena` — a parallel `Vec` of
/// nodes addressed by a public `ExprId`, with a raw slab of n-ary children —
/// was deleted in favour of it (`docs/plans/2026-09-09-exprarena-on-dag.md`);
/// a *new* graph should reach for this rather than hand-roll another `Vec`
/// plus index-newtype.
///
/// Only the consumption vocabulary — `Dag`, `Node`, `Rooted`, `Scratch`,
/// `SideTable` — is public at all. `Id`, `Key`, and `Builder` (which builds
/// a `Dag`) are `pub(crate)`: invisible outside this crate, not merely
/// unexported at the root. Memory management is a `Dag`'s own business —
/// `kernel.rs` and `expr.rs` build DAGs because they *are* this crate's
/// construction machinery; nothing further out ever needs to. What outside
/// callers build with is [`expr::ExprBuilder`], whose vocabulary is
/// expressions rather than nodes and edges.
pub mod dag;
pub use dag::{Dag, Node, Rooted, Scratch, SideTable};

pub mod expr;
pub use expr::{
    DecodeError, Environment, ExprBuilder, ExprData, ExprRef, Term, compute_dag_depth, decode,
    depth, display, encode, encode_into, has_degenerate, has_var, node_count_subtree, relink,
    retired_axis, subtree_eq,
};

/// IR-to-IR transforms: each takes an expression graph and returns another.
/// Target-blind by construction — nothing here knows which ISA it is feeding.
pub mod passes;

/// The term language the e-graph speaks: destructure a node, rebuild a node.
/// Naming it is what makes an optimizer expressible as an endomorphism on the
/// IR rather than as a hand-rolled conversion per tier.
pub mod term;
mod term_dag;
pub use term::{Children, Ir, Shape};
pub use term_dag::rebuild_into;

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

/// Fixture builders for this crate's own `benches/`/`tests/` binaries and
/// one `pixelflow-search` unit test — the only callers still allowed to
/// know a `Dag` is built at all. `pub` only because Cargo compiles a
/// `benches/`/`tests/` target as if it were an external crate, so there is
/// no visibility short of `pub` that reaches it; `#[doc(hidden)]` keeps it
/// out of rendered docs and out of the API this crate actually advertises.
/// Not for anything else. See the module doc for why.
#[doc(hidden)]
pub mod internal_test_support;
