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

/// Why [`emit::compile`](crate::emit::compile) and
/// [`jit_cache::compile`](crate::jit_cache::compile) can fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileError {
    /// [`pixelflow_ir::passes::legalize`] refused to lower part of the
    /// expression — e.g. differentiating a bound-memory read, or a construct
    /// with no derivative rule. The message names which; it is legalize's own
    /// message, carried here rather than restated, since this crate does not
    /// own that pass and a copy would drift from it.
    Legalize(&'static str),

    /// A fixed-size resource the register allocator or an emitted encoding is
    /// subject to — the 2MB spill frame, aarch64's 12-bit constant-pool `LDR`
    /// offset — was exceeded by this expression.
    BudgetExceeded(&'static str),

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
            Self::BudgetExceeded(msg) => write!(f, "compile budget exceeded: {msg}"),
            Self::Mmap => write!(f, "mmap failed"),
            Self::Mprotect => write!(f, "mprotect failed"),
        }
    }
}

impl core::error::Error for CompileError {}
