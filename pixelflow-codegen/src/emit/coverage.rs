//! The op-coverage completeness contract every [`super::IsaBackend`] must satisfy.
//!
//! This is test-only infrastructure (see `emit/mod.rs`'s `backend_op_coverage`
//! tests, and `x86_64.rs`'s `X86BinaryInsn::select` coverage test), not a
//! production API. It exists because nothing previously enumerated "the ops a
//! backend must support" anywhere: AVX-512's binary-op dispatch
//! (`avx512::emit_binary`) implemented 6 of the 15 required ops and nothing
//! caught the gap until 36 tests failed by accident the first time someone
//! actually compiled with `-C target-feature=+avx512f`. These lists turn that
//! into a named, itemized test failure instead of a silent hole.
//!
//! Every `OpKind` an `IsaBackend` might see falls into exactly one bucket:
//!
//! - Reaches `IsaBackend::emit_plan` as `ResolvedOp::Unary` — [`REQUIRED_UNARY_OPS`].
//! - Reaches it as `ResolvedOp::Binary` — [`REQUIRED_BINARY_OPS`].
//! - Reaches it as `ResolvedOp::ShiftImm` (the RHS `Const` shift amount is
//!   folded to an immediate by the DAG scheduler, so only the op + LHS
//!   survive to codegen) — [`REQUIRED_SHIFT_OPS`].
//! - Reaches it as `ResolvedOp::FusedMulAdd`/`DecomposedMulAdd` or
//!   `ResolvedOp::Select` — [`REQUIRED_TERNARY_OPS`]. Not swept generically
//!   like the arrays above (each has a distinct `ResolvedOp` shape and
//!   `setup_mov` convention), so the per-backend tests construct these two
//!   explicitly instead of looping.
//! - Is eliminated by a `lowering` pass before any backend sees it
//!   (transcendentals by `expand_transcendentals`, `Dwrt` by `lower_dwrt`,
//!   `Reduce` by `expand_reduce`, the ternary `Gather` op by `expand_gather`
//!   — see `lowering.rs`) — absent from every list here on purpose.
//! - Is structural and never becomes a `ScheduledOp` at all (`Var`, `Const`,
//!   `Tuple`, `Buffer`) — likewise absent.
//! - `RawGather` (bound-memory read) reaches `ResolvedOp::Gather`, but is
//!   intentionally backend-asymmetric today (native `vgatherdps` on AVX-512;
//!   a four-lane scalar-load sequence on SSE2/aarch64) — deliberately NOT
//!   included in a "every backend must support this" list; its own test
//!   coverage lives with the gather-specific tests.
//! - `Uniform` (per-call scalar) reaches `ResolvedOp::Uniform` as a leaf
//!   definition with no operands — a broadcast load from the block — and is
//!   pinned byte-for-byte per backend by `tests::uniforms` in `mod.rs`.

use pixelflow_ir::kind::OpKind;

/// Ops that reach `IsaBackend::emit_plan` as `ResolvedOp::Unary { op, .. }`.
pub(crate) const REQUIRED_UNARY_OPS: &[OpKind] = &[
    OpKind::Neg,
    OpKind::Sqrt,
    OpKind::Rsqrt,
    OpKind::Abs,
    OpKind::Recip,
    OpKind::Floor,
    OpKind::Ceil,
    OpKind::Round,
    OpKind::TruncToInt,
    OpKind::IntToFloat,
];

/// Ops that reach `IsaBackend::emit_plan` as `ResolvedOp::Binary { op, .. }`.
pub(crate) const REQUIRED_BINARY_OPS: &[OpKind] = &[
    OpKind::Add,
    OpKind::Sub,
    OpKind::Mul,
    OpKind::Div,
    OpKind::Min,
    OpKind::Max,
    OpKind::Lt,
    OpKind::Le,
    OpKind::Gt,
    OpKind::Ge,
    OpKind::Eq,
    OpKind::Ne,
    OpKind::IAdd,
    OpKind::BitAnd,
    OpKind::BitOr,
];

/// Ops that reach `IsaBackend::emit_plan` as `ResolvedOp::ShiftImm { op, .. }`.
pub(crate) const REQUIRED_SHIFT_OPS: &[OpKind] = &[OpKind::Shl, OpKind::Shr];

/// Ops with a bespoke `ResolvedOp` shape (`FusedMulAdd`/`DecomposedMulAdd` for
/// `MulAdd`, `Select` for `Select`). Listed for documentation; the per-backend
/// tests build these plans explicitly rather than looping generically — four
/// of them for `MulAdd` alone, since a backend owes both shapes and each
/// `DeferredReload` spelling of the decomposed one is its own arm.
pub(crate) const REQUIRED_TERNARY_OPS: &[OpKind] = &[OpKind::MulAdd, OpKind::Select];
