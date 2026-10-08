//! The op-coverage completeness contract every [`super::IsaBackend`] must satisfy.
//!
//! This is test-only infrastructure (see `emit/mod.rs`'s `backend_op_coverage`
//! tests), not a production API. It exists because nothing previously enumerated "the ops a
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
//!   folded to an immediate by `program::lower::arena_to_schedule`, so only the op + LHS
//!   survive to codegen) — [`REQUIRED_SHIFT_OPS`].
//! - Reaches it as `ResolvedOp::FusedMulAdd`/`DecomposedMulAdd` or
//!   `ResolvedOp::If` — [`REQUIRED_TERNARY_OPS`]. Not swept generically
//!   like the arrays above (each has a distinct `ResolvedOp` shape and
//!   `setup_mov` convention), so the per-backend tests construct these two
//!   explicitly instead of looping.
//! - Is eliminated by a `lowering` pass before any backend sees it
//!   (transcendentals by `expand_transcendentals`, `Dwrt` by `lower_dwrt`,
//!   the ternary `Gather` op by `expand_gather` — see `lowering.rs`) — absent
//!   from every list here on purpose.
//! - Is a loop, not an op: a `Reduce` survives `legalize` and opens a
//!   `regalloc::Scope::Fold` — likewise absent.
//! - Is structural and never becomes a `ScheduledOp` at all (`Var`, `Const`,
//!   `Tuple`, `Buffer`) — likewise absent.
//! - `RawGather` (bound-memory read) reaches `ResolvedOp::Gather`, but is
//!   intentionally backend-asymmetric today (native `vgatherdps` on AVX2 and
//!   AVX-512; a four-lane scalar-load sequence on aarch64) — deliberately NOT
//!   included in a "every backend must support this" list; its own test
//!   coverage lives with the gather-specific tests. A `RawGather` whose
//!   index the lane binder does not reach is `ResolvedOp::Broadcast`
//!   instead — one scalar load broadcast, the same shape on every backend —
//!   pinned byte-for-byte per backend by `tests::broadcast` in `mod.rs`.
//! - `Uniform` (per-call scalar) reaches `ResolvedOp::Uniform` as a leaf
//!   definition with no operands — a broadcast load from the block — and is
//!   pinned byte-for-byte per backend by `tests::uniforms` in `mod.rs`.
//!
//! "Absent from every list on purpose" is checked, not asserted: the test at the
//! foot of this file walks `OpKind::all()` and demands each op sit in exactly
//! one list above or be named, with its reason, in the test-local `NOT_REQUIRED`.

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
/// `MulAdd`, `If` for `If`). Listed for documentation; the per-backend
/// tests build these plans explicitly rather than looping generically — four
/// of them for `MulAdd` alone, since a backend owes both shapes and each
/// `DeferredReload` spelling of the decomposed one is its own arm.
pub(crate) const REQUIRED_TERNARY_OPS: &[OpKind] = &[OpKind::MulAdd, OpKind::If];

/// Ops no `REQUIRED_*` list holds, each for the reason beside it. The test
/// below is the only reader.
const NOT_REQUIRED: &[OpKind] = &[
    // Lowered away by `legalize`, so no backend ever sees one: the
    // transcendentals by `expand_transcendentals`,
    OpKind::Sin,
    OpKind::Cos,
    OpKind::Tan,
    OpKind::Asin,
    OpKind::Acos,
    OpKind::Atan,
    OpKind::Atan2,
    OpKind::Exp,
    OpKind::Exp2,
    OpKind::Ln,
    OpKind::Log2,
    OpKind::Log10,
    OpKind::Pow,
    // `Dwrt` by `lower_dwrt`,
    OpKind::Dwrt,
    // and `Gather` by `expand_gather`, into index arithmetic plus `RawGather`.
    OpKind::Gather,
    // Leaves and memory: each reaches a backend, if at all, as a `ResolvedOp`
    // of its own shape, pinned by its own test rather than swept.
    OpKind::Var,       // a binder's placeholder (`Nop`), or the lane iota (`Lanes`)
    OpKind::Const,     // `LoadConst`
    OpKind::Buffer,    // `Context`, a base pointer
    OpKind::Uniform,   // `Uniform`, a broadcast load from the block
    OpKind::RawGather, // `Gather`, or `Broadcast` when the index is lane-uniform
    // Never an instruction: handled by the scope walker, or refused outright.
    OpKind::Reduce, // opens a `Scope::Fold`, a loop
    OpKind::Seq,    // an effect; emits no bytes
    OpKind::Tuple,  // `Nary`, which `arena_to_schedule` refuses
    OpKind::Param,  // a macro-tier slot, refused at scheduling
];

#[test]
fn every_op_is_in_exactly_one_required_list_or_named_as_not_required() {
    let lists: [&[OpKind]; 5] = [
        REQUIRED_UNARY_OPS,
        REQUIRED_BINARY_OPS,
        REQUIRED_SHIFT_OPS,
        REQUIRED_TERNARY_OPS,
        NOT_REQUIRED,
    ];

    let misplaced: Vec<String> = OpKind::all()
        .filter_map(|op| {
            let listed: usize = lists
                .iter()
                .map(|list| list.iter().filter(|&&listed| listed == op).count())
                .sum();
            (listed != 1).then(|| format!("{op:?} is listed {listed} times"))
        })
        .collect();

    assert!(
        misplaced.is_empty(),
        "every op must be listed exactly once across the four REQUIRED_* lists \
         and NOT_REQUIRED: {misplaced:?}. A new op the backends must encode \
         belongs in a REQUIRED_* list; one that never reaches them belongs in \
         NOT_REQUIRED, with its reason."
    );
}
