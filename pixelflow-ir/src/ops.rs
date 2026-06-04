//! Concrete Operation Structs (ZSTs).
//!
//! Each struct represents a specific operation in the IR as a Zero-Sized Type.
//! These implement the `Op` trait with type-level constants.
//!
//! The `ALL_OPS` array is the single source of truth for all operations.

use crate::traits::{Arity, EmitStyle, Op, OpMeta};

macro_rules! define_op {
    ($idx:expr, $name:ident, $str_name:expr, $arity:expr, $emit:expr) => {
        #[doc = $str_name]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name;

        impl Arity for $name {
            const ARITY: usize = $arity;
        }

        impl OpMeta for $name {
            #[inline(always)]
            fn name(&self) -> &'static str {
                $str_name
            }
            #[inline(always)]
            fn emit_style(&self) -> EmitStyle {
                $emit
            }
            #[inline(always)]
            fn index(&self) -> usize {
                $idx
            }
            #[inline(always)]
            fn arity(&self) -> usize {
                $arity
            }
        }

        impl Op for $name {
            const NAME: &'static str = $str_name;
            const EMIT_STYLE: EmitStyle = $emit;
            const INDEX: usize = $idx;
        }
    };
}

use EmitStyle::*;

// --- Basic Arithmetic ---
define_op!(0, Var, "var", 0, Special);
define_op!(1, Const, "const", 0, Special);
define_op!(2, Add, "add", 2, BinaryInfix("+"));
define_op!(3, Sub, "sub", 2, BinaryInfix("-"));
define_op!(4, Mul, "mul", 2, BinaryInfix("*"));
define_op!(5, Div, "div", 2, BinaryInfix("/"));
define_op!(6, Neg, "neg", 1, UnaryPrefix);
define_op!(7, Sqrt, "sqrt", 1, UnaryMethod);
define_op!(8, Rsqrt, "rsqrt", 1, UnaryMethod);
define_op!(9, Abs, "abs", 1, UnaryMethod);
define_op!(10, Min, "min", 2, BinaryMethod);
define_op!(11, Max, "max", 2, BinaryMethod);
define_op!(12, MulAdd, "mul_add", 3, TernaryMethod);

// --- Extended Math ---
define_op!(13, Recip, "recip", 1, UnaryMethod);
define_op!(14, Floor, "floor", 1, UnaryMethod);
define_op!(15, Ceil, "ceil", 1, UnaryMethod);
define_op!(16, Round, "round", 1, UnaryMethod);
define_op!(17, Fract, "fract", 1, UnaryMethod);

// --- Trigonometry ---
define_op!(18, Sin, "sin", 1, UnaryMethod);
define_op!(19, Cos, "cos", 1, UnaryMethod);
define_op!(20, Tan, "tan", 1, UnaryMethod);
define_op!(21, Asin, "asin", 1, UnaryMethod);
define_op!(22, Acos, "acos", 1, UnaryMethod);
define_op!(23, Atan, "atan", 1, UnaryMethod);
define_op!(24, Atan2, "atan2", 2, BinaryMethod);

// --- Exponentials ---
define_op!(25, Exp, "exp", 1, UnaryMethod);
define_op!(26, Exp2, "exp2", 1, UnaryMethod);
define_op!(27, Ln, "ln", 1, UnaryMethod);
define_op!(28, Log2, "log2", 1, UnaryMethod);
define_op!(29, Log10, "log10", 1, UnaryMethod);
define_op!(30, Pow, "pow", 2, BinaryMethod);
define_op!(31, Hypot, "hypot", 2, BinaryMethod);

// --- Comparison ---
define_op!(32, Lt, "lt", 2, BinaryMethod);
define_op!(33, Le, "le", 2, BinaryMethod);
define_op!(34, Gt, "gt", 2, BinaryMethod);
define_op!(35, Ge, "ge", 2, BinaryMethod);
define_op!(36, Eq, "eq", 2, BinaryMethod);
define_op!(37, Ne, "ne", 2, BinaryMethod);

// --- Control Flow ---
define_op!(38, Select, "select", 3, TernaryMethod);
define_op!(39, Clamp, "clamp", 3, TernaryMethod);

// --- Structure ---
define_op!(40, Tuple, "tuple", 0, Special);

// Bit-manipulation primitives (integer-domain). Each maps 1:1 to a single
// instruction; they let exp/ln/log lower to arithmetic.
define_op!(41, TruncToInt, "trunc_to_int", 1, UnaryMethod);
define_op!(42, IntToFloat, "int_to_float", 1, UnaryMethod);
define_op!(43, IAdd, "iadd", 2, BinaryMethod);
define_op!(44, Shl, "shl", 2, BinaryMethod);
define_op!(45, Shr, "shr", 2, BinaryMethod);
define_op!(46, BitAnd, "bitand", 2, BinaryMethod);
define_op!(47, BitOr, "bitor", 2, BinaryMethod);

// Differentiation operator. Rewritten away in the e-graph (chain rule); never
// emitted, hence Special.
define_op!(48, Dwrt, "dwrt", 2, Special);

/// Total number of operations. Must equal [`crate::kind::OpKind::COUNT`].
pub const OP_COUNT: usize = 49;

/// All operations in the IR, indexed by their INDEX constant.
///
/// This is the single source of truth for all operations.
pub const ALL_OPS: [&'static dyn OpMeta; OP_COUNT] = [
    &Var, &Const, &Add, &Sub, &Mul, &Div, &Neg, &Sqrt, &Rsqrt, &Abs, &Min, &Max, &MulAdd, &Recip,
    &Floor, &Ceil, &Round, &Fract, &Sin, &Cos, &Tan, &Asin, &Acos, &Atan, &Atan2, &Exp, &Exp2, &Ln,
    &Log2, &Log10, &Pow, &Hypot, &Lt, &Le, &Gt, &Ge, &Eq, &Ne, &Select, &Clamp, &Tuple,
    &TruncToInt, &IntToFloat, &IAdd, &Shl, &Shr, &BitAnd, &BitOr, &Dwrt,
];

// Compile-time guard: the two op counts must agree.
const _: () = assert!(OP_COUNT == crate::kind::OpKind::COUNT);

/// Get an operation by name.
pub fn op_by_name(name: &str) -> Option<&'static dyn OpMeta> {
    ALL_OPS.iter().find(|op| op.name() == name).copied()
}

/// Get an operation by index.
pub fn op_by_index(idx: usize) -> Option<&'static dyn OpMeta> {
    ALL_OPS.get(idx).copied()
}

/// All method names that can be used in kernel! expressions.
///
/// Derived from ALL_OPS, excluding Special emit style.
pub fn known_method_names() -> impl Iterator<Item = &'static str> {
    ALL_OPS
        .iter()
        .filter(|op| !matches!(op.emit_style(), Special))
        .map(|op| op.name())
}
