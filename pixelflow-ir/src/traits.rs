//! Core traits for the Intermediate Representation.

use core::fmt::Debug;
use core::hash::Hash;

/// How an operation should be emitted in generated code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmitStyle {
    /// Not directly emittable (Var, Const handled specially)
    Special,
    /// Unary prefix: `(-a)`
    UnaryPrefix,
    /// Unary method: `(a).sqrt()`
    UnaryMethod,
    /// Binary infix: `(a + b)`
    BinaryInfix(&'static str),
    /// Binary method: `(a).min(b)`
    BinaryMethod,
    /// Binary method with different Rust name: `(a).powf(b)` for pow
    BinaryMethodNamed(&'static str),
    /// Ternary method: `(a).clamp(b, c)`
    TernaryMethod,
}

/// Base trait for operation arity.
pub trait Arity {
    /// Number of operands.
    const ARITY: usize;
}

/// Marker trait for nullary operations (0 operands).
pub trait Nullary: Arity {}

/// Marker trait for unary operations (1 operand).
pub trait Unary: Arity {}

/// Marker trait for binary operations (2 operands).
pub trait Binary: Arity {}

/// Marker trait for ternary operations (3 operands).
pub trait Ternary: Arity {}

/// Marker trait for variadic/n-ary operations.
pub trait Nary: Arity {}

/// Dynamic-dispatch-compatible operation metadata.
///
/// This trait contains only the methods needed for codegen and lookup.
/// It's dyn-compatible because it doesn't use Self in bounds.
pub trait OpMeta: 'static + Debug + Send + Sync {
    /// Display name of the operation (e.g., "sqrt", "add").
    fn name(&self) -> &'static str;

    /// How to emit this operation in generated code.
    fn emit_style(&self) -> EmitStyle;

    /// Unique index for this operation (for feature vectors, etc.)
    fn index(&self) -> usize;

    /// Number of operands.
    fn arity(&self) -> usize;
}

/// A static operation in the IR.
///
/// This trait is implemented by ZSTs (unit structs) representing individual
/// operations. It defines the ISA properties at the type level.
///
/// Also implements `OpMeta` for dyn-compatible access.
pub trait Op: 'static + Arity + Eq + Hash + Copy + Clone + Debug + Send + Sync + OpMeta {
    /// Display name of the operation (e.g., "sqrt", "add").
    const NAME: &'static str;

    /// How to emit this operation in generated code.
    const EMIT_STYLE: EmitStyle;

    /// Unique index for this operation (for feature vectors, etc.)
    const INDEX: usize;
}
