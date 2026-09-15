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

// === Comparison ===
define_op!(Lt);
define_op!(Le);
define_op!(Gt);
define_op!(Ge);
define_op!(Eq);
define_op!(Ne);

// === Control Flow ===
define_op!(Select);

// === Aggregates ===
define_op!(Tuple);

// === Memory ===
// `Gather(buffer, x, y)` — bound-memory read (floor indices, clamp to the
// declared extents, row-major). Present as an `Op` so the e-graph can carry it
// as opaque structure — two gathers hash-cons into one e-class iff buffer
// identity and all three children match — but deliberately ABSENT from
// [`op_from_kind`]: rewrite templates resolve ops through that lookup, and
// returning `None` there is what keeps the rule set unable to name Gather.
// Its semantics depend on bound memory, so no algebraic rule (and no constant
// fold) may ever look through it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Gather;

impl Op for Gather {
    #[inline]
    fn kind(&self) -> OpKind {
        OpKind::Gather
    }
}

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
        OpKind::Lt => Some(&Lt),
        OpKind::Le => Some(&Le),
        OpKind::Gt => Some(&Gt),
        OpKind::Ge => Some(&Ge),
        OpKind::Eq => Some(&Eq),
        OpKind::Ne => Some(&Ne),
        OpKind::Select => Some(&Select),
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
        // Memory ops are representable (`ENode::Buffer` carries the full
        // `BufferDecl` — `BufferIdentity` is process-unique, so no arena-local
        // table is needed) but stay opaque to the rule set: `None` here is
        // what denies every rewrite template the ability to name them, so
        // their participation is hash-consing CSE only. `add_arena` and the
        // runtime tier resolve `Gather` directly via the crate-private
        // `Gather` op above. `RawGather` and `Reduce` are lowered
        // before/after the e-graph and never appear in one.
        // `Uniform` and `Param` likewise: leaves the e-graph holds as
        // `ENode::Uniform`/`ENode::Param`, never as an op a rule could match —
        // which is exactly what keeps `ConstantFold` from ever seeing one.
        OpKind::Buffer
        | OpKind::Gather
        | OpKind::RawGather
        | OpKind::Reduce
        | OpKind::Uniform
        | OpKind::Param => None,
    }
}

// ─────────────────────────── Runtime-only mask ops ───────────────────────────
//
// `&`/`|` on comparison masks are surface-language ops (every glyph winding
// kernel's Y-range gate is `(Y >= lo) & (Y < hi)`), so runtime-built arenas
// must be representable with them present. They are deliberately NOT in
// [`op_from_kind`]: registering them globally hands them to the
// AOT macro tier too, whose e-graph runs at macro-expansion time — BEFORE
// composition — where resolving the `Dwrt` nodes the masks travel with is
// unsound (a leaf's `DX` is 1 only until an enclosing `.at()` warp scales
// it; the fonts' density-dependent AA ramp broke exactly this way when
// these ops were briefly global). The runtime tier optimizes the final
// composed arena at bake time, where the calculus has its full context, so
// mask ops are safe — and only meaningful — here.
//
// No rewrite rule targets them; they participate as opaque structure plus
// `ConstantFold`, whose bitwise-domain exemption already models their
// all-ones/zero masks.
struct MaskAnd;
impl Op for MaskAnd {
    fn kind(&self) -> OpKind {
        OpKind::BitAnd
    }
}
struct MaskOr;
impl Op for MaskOr {
    fn kind(&self) -> OpKind {
        OpKind::BitOr
    }
}

/// Mask `∧` as an `Op`, for a caller that already named the *algebra*.
///
/// Crate-visible where [`op_from_kind`] is not: resolving an `OpKind` to
/// these would hand them to the macro tier, which is the miscompile the
/// comment above describes. Resolving a [`Monoid`](pixelflow_ir::Monoid) to
/// them leaks no opcode — the caller said "the universal quantifier", and
/// this is what that is.
pub(crate) fn mask_and() -> &'static dyn Op {
    &MaskAnd
}

/// Mask `∨` as an `Op`. See [`mask_and`].
pub(crate) fn mask_or() -> &'static dyn Op {
    &MaskOr
}

// ─────────────────────── Runtime-only integer-domain ops ─────────────────────
//
// The packed cell-grid kernel's spine: clamp → `TruncToInt` → `Shl` →
// or-fold builds a `u32` pixel per lane, so the production frame kernel is
// unrepresentable — and therefore compiles with NO CSE across its four
// channels — unless these enter the e-graph. Runtime-tier only, for the
// same reason as the mask ops above. Opaque to TEMPLATES: no rewrite rule can
// name them (nothing here or in `op_from_kind` hands them to a template), and
// their results are bit patterns the float rule set has no semantics for.
//
// Template-opacity is NOT fold-opacity, and the distinction is load-bearing:
// `ConstantFold::apply` destructures any `ENode::Op` and reads `op.kind()`
// (`math::algebra`) — it never consults `op_from_kind`. So every op registered
// here folds, and each one needs its own answer to "does this fold agree with
// what the backends emit?" `OpKind::fold_is_platform_specific` is where that
// answer lives; being unnameable by a template guards nothing.
//
// `Shl`/`Shr` do keep `Const` shift operands, because extraction emits `Const`
// leaves verbatim — so the emitter's immediate-only contract holds. The count's
// RANGE is a separate matter, enforced where the `Const` narrows to an
// immediate (`emit::shift_immediate`) rather than assumed here.
struct IntTrunc;
impl Op for IntTrunc {
    fn kind(&self) -> OpKind {
        OpKind::TruncToInt
    }
}
struct IntFromInt;
impl Op for IntFromInt {
    fn kind(&self) -> OpKind {
        OpKind::IntToFloat
    }
}
struct IntAdd;
impl Op for IntAdd {
    fn kind(&self) -> OpKind {
        OpKind::IAdd
    }
}
struct IntShl;
impl Op for IntShl {
    fn kind(&self) -> OpKind {
        OpKind::Shl
    }
}
struct IntShr;
impl Op for IntShr {
    fn kind(&self) -> OpKind {
        OpKind::Shr
    }
}

/// Which ops an e-graph is allowed to *hold*, as opposed to which ops a
/// rewrite template may *name* ([`op_from_kind`], always the smaller set).
///
/// The two differ, and the difference is per-tier rather than global — which
/// is the whole reason this is an argument. Mask and integer-domain ops are
/// sound in a runtime-tier graph (built at bake time, after composition, where
/// the derivative calculus has its full context) and unsound in a macro-tier
/// one (expanded before composition, where a leaf's `DX` is not yet scaled by
/// an enclosing warp). That was previously encoded in *which private helper a
/// call site happened to call*, so nothing checked it and the two helpers had
/// drifted apart on five op kinds besides. Naming the vocabulary makes the
/// choice visible at each insertion, and wrong only by writing the wrong
/// variant.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Vocabulary {
    /// What a rewrite template can name, plus the opaque `Gather`. The
    /// macro (`kernel!`) and `Dwrt` expansion tiers.
    Templates,
    /// [`Self::Templates`] plus the mask and integer-domain ops above. The
    /// runtime tier, which optimizes composed arenas at bake time.
    Runtime,
}

impl Vocabulary {
    /// The static `Op` for `kind` in this vocabulary, or `None` if a graph
    /// under it may not hold that op.
    #[must_use]
    pub fn resolve(self, kind: OpKind) -> Option<&'static dyn Op> {
        // Opaque in both: representable as structure (hash-consing CSE), never
        // nameable by a template — `op_from_kind` returns `None` for it.
        if kind == OpKind::Gather {
            return Some(&Gather);
        }
        if self == Self::Runtime {
            match kind {
                OpKind::BitAnd => return Some(&MaskAnd),
                OpKind::BitOr => return Some(&MaskOr),
                OpKind::TruncToInt => return Some(&IntTrunc),
                OpKind::IntToFloat => return Some(&IntFromInt),
                OpKind::IAdd => return Some(&IntAdd),
                OpKind::Shl => return Some(&IntShl),
                OpKind::Shr => return Some(&IntShr),
                _ => {}
            }
        }
        op_from_kind(kind)
    }
}
