//! Operation definitions for e-graph expressions.
//!
//! Each operation is a unit struct implementing the `Op` trait.
//! Properties are delegated to `OpKind` - the single source of truth.

use pixelflow_ir::{EmitStyle, OpKind};

/// Trait for operations in the e-graph.
///
/// All algebraic properties delegate to `OpKind` methods.
pub trait Op: 'static + Send + Sync {
    /// The canonical `OpKind` for this operation.
    fn kind(&self) -> OpKind;

    /// String name (delegates to `OpKind::name`).
    #[inline]
    fn name(&self) -> &'static str {
        self.kind().name()
    }

    /// How to emit this operation in generated code.
    #[inline]
    fn emit_style(&self) -> EmitStyle {
        self.kind().emit_style()
    }

    /// Default cost estimate (delegates to `OpKind::default_cost`).
    #[inline]
    fn default_cost(&self) -> usize {
        self.kind().default_cost()
    }

    /// Commutativity (delegates to `OpKind::is_commutative`).
    #[inline]
    fn is_commutative(&self) -> bool {
        self.kind().is_commutative()
    }

    /// Associativity (delegates to `OpKind::is_associative`).
    #[inline]
    fn is_associative(&self) -> bool {
        self.kind().is_associative()
    }

    /// Identity element (delegates to `OpKind::identity`).
    #[inline]
    fn identity(&self) -> Option<f32> {
        self.kind().identity()
    }

    /// Annihilator element (delegates to `OpKind::annihilator`).
    #[inline]
    fn annihilator(&self) -> Option<f32> {
        self.kind().annihilator()
    }

    /// Idempotency (delegates to `OpKind::is_idempotent`).
    #[inline]
    fn is_idempotent(&self) -> bool {
        self.kind().is_idempotent()
    }
}

impl core::fmt::Debug for dyn Op {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Op({})", self.name())
    }
}

/// Generate a ZST operation struct that delegates to `OpKind`.
macro_rules! define_op {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name;

        impl Op for $name {
            #[inline]
            fn kind(&self) -> OpKind {
                OpKind::$name
            }
        }
    };
}

// === Basic Arithmetic ===
define_op!(Add);
define_op!(Sub);
define_op!(Mul);
define_op!(Div);
define_op!(Neg);
define_op!(Recip);

// === Roots ===
define_op!(Sqrt);
define_op!(Rsqrt);

// === Misc Math ===
define_op!(Abs);
define_op!(Min);
define_op!(Max);
define_op!(MulAdd);

// === Rounding ===
define_op!(Floor);
define_op!(Ceil);
define_op!(Round);
define_op!(Fract);

// === Trigonometry ===
define_op!(Sin);
define_op!(Cos);
define_op!(Tan);
define_op!(Asin);
define_op!(Acos);
define_op!(Atan);
define_op!(Atan2);

// === Exponentials/Logarithms ===
define_op!(Exp);
define_op!(Exp2);
define_op!(Ln);
define_op!(Log2);
define_op!(Log10);
define_op!(Pow);
define_op!(Hypot);

// === Comparison ===
define_op!(Lt);
define_op!(Le);
define_op!(Gt);
define_op!(Ge);
define_op!(Eq);
define_op!(Ne);

// === Control Flow ===
define_op!(Select);
define_op!(Clamp);

// === Aggregates ===
define_op!(Tuple);

// === Differentiation ===
// `Dwrt(expr, var)` is the single autodiff operator. It exists only inside the
// e-graph: chain-rule rewrites push it toward the leaves until it dissolves
// into ordinary arithmetic. A surviving `Dwrt` after saturation is the jet
// fallback (not yet wired) and carries a prohibitive cost so the extractor
// never prefers it.
define_op!(Dwrt);

/// Look up a static `&dyn Op` reference by `OpKind`.
///
/// Returns `None` for `Var` and `Const` (which are leaves, not operations).
pub fn op_from_kind(kind: OpKind) -> Option<&'static dyn Op> {
    match kind {
        OpKind::Add => Some(&Add),
        OpKind::Sub => Some(&Sub),
        OpKind::Mul => Some(&Mul),
        OpKind::Div => Some(&Div),
        OpKind::Neg => Some(&Neg),
        OpKind::Recip => Some(&Recip),
        OpKind::Sqrt => Some(&Sqrt),
        OpKind::Rsqrt => Some(&Rsqrt),
        OpKind::Abs => Some(&Abs),
        OpKind::Min => Some(&Min),
        OpKind::Max => Some(&Max),
        OpKind::MulAdd => Some(&MulAdd),
        OpKind::Floor => Some(&Floor),
        OpKind::Ceil => Some(&Ceil),
        OpKind::Round => Some(&Round),
        OpKind::Fract => Some(&Fract),
        OpKind::Sin => Some(&Sin),
        OpKind::Cos => Some(&Cos),
        OpKind::Tan => Some(&Tan),
        OpKind::Asin => Some(&Asin),
        OpKind::Acos => Some(&Acos),
        OpKind::Atan => Some(&Atan),
        OpKind::Atan2 => Some(&Atan2),
        OpKind::Exp => Some(&Exp),
        OpKind::Exp2 => Some(&Exp2),
        OpKind::Ln => Some(&Ln),
        OpKind::Log2 => Some(&Log2),
        OpKind::Log10 => Some(&Log10),
        OpKind::Pow => Some(&Pow),
        OpKind::Hypot => Some(&Hypot),
        OpKind::Lt => Some(&Lt),
        OpKind::Le => Some(&Le),
        OpKind::Gt => Some(&Gt),
        OpKind::Ge => Some(&Ge),
        OpKind::Eq => Some(&Eq),
        OpKind::Ne => Some(&Ne),
        OpKind::Select => Some(&Select),
        OpKind::Clamp => Some(&Clamp),
        OpKind::Tuple => Some(&Tuple),
        // Autodiff operator: lives in the e-graph, rewritten by the chain rule.
        OpKind::Dwrt => Some(&Dwrt),
        // Leaves (not operations)
        OpKind::Var | OpKind::Const => None,
        // Bit-manip primitives are produced by *lowering* (after the e-graph
        // runs), so they have no rewrite-rule `Op` and never appear in an
        // e-graph. Treated as opaque here.
        OpKind::TruncToInt
        | OpKind::IntToFloat
        | OpKind::IAdd
        | OpKind::Shl
        | OpKind::Shr
        | OpKind::BitAnd
        | OpKind::BitOr => None,
    }
}
