//! Why compiling an expression DAG to machine code failed.
//!
//! A `&'static str` answers none of the questions a caller actually has:
//! "should I retry with a smaller kernel", "is this a bug in `pixelflow-codegen`
//! I should report", "did the OS just refuse me memory" are three different
//! responses, and a message string makes a caller `matches!` on text to tell
//! them apart. [`CompileError`] denotes the ways a compile fails as variants
//! instead — one family per real `Err` site in this crate, grouped by what a
//! caller would actually do differently in response.

use core::fmt;

use pixelflow_ir::kind::OpKind;

/// Why [`emit::compile_arena`](crate::emit::compile_arena),
/// [`emit::compile_collapse`](crate::emit::compile_collapse), and the rest of
/// this crate's compile entries can fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileError {
    /// [`pixelflow_ir::passes::legalize`] refused to lower part of the
    /// expression — e.g. differentiating a bound-memory read, or a construct
    /// with no derivative rule. The message names which; it is legalize's own
    /// message, carried here rather than restated, since this crate does not
    /// own that pass and a copy would drift from it.
    Legalize(&'static str),

    /// An op reached ternary position in the DAG emitter that isn't `Select`
    /// or `MulAdd`. `passes::legalize` is supposed to leave only those two in
    /// that position, so this is the pipeline having let one through rather
    /// than a fact about `op`.
    UnsupportedOp(OpKind),

    /// A fixed-size resource the register allocator or an emitted encoding is
    /// subject to — the 2MB spill frame, aarch64's 12-bit constant-pool `LDR`
    /// offset — was exceeded by this expression.
    BudgetExceeded(&'static str),

    /// This crate's own bookkeeping (frame layout, red-zone addressing)
    /// caught itself in a state that should be unreachable for any input.
    /// Always a bug in `pixelflow-codegen`, never a fact about the kernel.
    Internal(&'static str),

    /// A branch named a [`Label`](crate::emit::Label) that no
    /// [`Item::Bind`](crate::emit::Item::Bind) ever bound.
    ///
    /// An error rather than a panic because it is the one thing a *program*
    /// can get wrong that the type system does not already refuse: minting is
    /// separate from binding precisely so a forward branch can name a position
    /// that does not exist yet, which means "never bound" is representable.
    UnboundLabel,

    /// One label was bound at two positions. A name that means two places is
    /// not a name.
    DuplicateLabel,

    /// A branch's displacement does not fit its encoding's field — an aarch64
    /// `B` reaches ±128 MiB and a `B.cond` only ±1 MiB.
    ///
    /// A real limit of the instruction, not an invariant to assume away: a
    /// fully unrolled fold is exactly the kind of body that grows until a
    /// conditional branch can no longer span it.
    BranchOutOfRange,

    /// [`ExecutableCode::from_code`](crate::emit::executable::ExecutableCode::from_code)
    /// was given an empty code buffer — nothing to map or execute.
    EmptyCodeBuffer,

    /// `mmap` refused to create the read-write staging mapping for the
    /// compiled code.
    Mmap,

    /// `mprotect` refused to flip the staging mapping to read-execute (W^X).
    Mprotect,
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Legalize(msg) => write!(f, "expression cannot be legalized: {msg}"),
            Self::UnsupportedOp(op) => {
                write!(f, "unsupported ternary op in DAG compilation: {op:?}")
            }
            Self::BudgetExceeded(msg) => write!(f, "compile budget exceeded: {msg}"),
            Self::Internal(msg) => {
                write!(f, "internal pixelflow-codegen invariant violated: {msg}")
            }
            Self::EmptyCodeBuffer => write!(f, "empty code buffer"),
            Self::Mmap => write!(f, "mmap failed"),
            Self::Mprotect => write!(f, "mprotect failed"),
            Self::UnboundLabel => write!(f, "a branch names a label nothing bound"),
            Self::DuplicateLabel => write!(f, "a label was bound at two positions"),
            Self::BranchOutOfRange => {
                write!(f, "branch displacement does not fit its encoding")
            }
        }
    }
}

impl core::error::Error for CompileError {}
