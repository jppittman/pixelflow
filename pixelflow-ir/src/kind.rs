//! Operation Kind Enumeration.
//!
//! This enum provides a uniform representation of all operations.
//! It is used for storage in the e-graph and as the base for feature indices.
//!
//! Per-op tables belong in [`OpMap`], not in a bare `[T; OpKind::COUNT]`.
//! A bare array makes each consumer responsible for turning an op into a
//! subscript, and every consumer that did got it wrong the same way.

use crate::traits::EmitStyle;
use core::ops::{Index, IndexMut};

/// The op table: the single place an operation is declared.
///
/// The enum, the roster, the count and both directions of the numbering are
/// all generated from this one list, so they cannot disagree and none of them
/// can be forgotten. Adding an operation is adding a line here; there is no
/// second place to update and therefore no second place to get wrong.
///
/// A macro rather than four declarations kept in step by a check, because a
/// check can only visit the variants something hands it: a variant missing
/// from the roster is exactly the one it never sees, and its absence is
/// therefore invisible. Generating all four removes that possibility instead
/// of testing for it.
///
/// The designs this replaced, and what each got wrong, are in
/// `docs/designs/opkind-numbering-is-private.md` §4.
macro_rules! op_table {
    ($( $(#[$attr:meta])* $name:ident = $code:literal, )+) => {
        /// Unified enumeration of all IR operations.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(u8)]
        pub enum OpKind {
            $( $(#[$attr])* $name = $code, )+
        }

        impl OpKind {
            /// Total number of operations.
            ///
            /// Generated from the table, so it grows with it.
            pub(crate) const COUNT: usize = [$( stringify!($name), )+].len();

            /// Every op, in [`OpKind::index`] order.
            ///
            /// Generated from the table; [`OpKind::all`] is the public way in.
            pub(crate) const ALL: [Self; Self::COUNT] = [$( Self::$name, )+];

            /// This op's slot number — the subscript [`OpMap`] uses, and
            /// nothing else.
            ///
            /// Crate-private on purpose. It is not an opcode, not a wire byte,
            /// and carries no promise outside: it exists so a per-op table has
            /// somewhere to put things. Code that needs an op in bytes wants
            /// [`OpKind::marshal`], whose encoding this crate may change.
            #[inline]
            #[must_use]
            pub(crate) const fn index(self) -> usize {
                match self { $( Self::$name => $code, )+ }
            }

            /// Inverse of [`OpKind::index`]. `None` for any integer that names
            /// no op — no `unsafe`, so an unnamed integer cannot be conjured
            /// into a variant that does not exist.
            #[must_use]
            pub(crate) const fn from_index(idx: usize) -> Option<Self> {
                match idx { $( $code => Some(Self::$name), )+ _ => None }
            }

            /// This op's Rust variant identifier — [`OpKind::MulAdd`]'s is
            /// `"MulAdd"`.
            ///
            /// Distinct from [`OpKind::name`], which is the DSL spelling
            /// (`"mul_add"`). A code generator wants the identifier: emitting
            /// the path `OpKind::#variant` is checked by the compiler at the
            /// expansion's use site, where emitting a `from_name` call would
            /// defer a typo to a runtime `expect` inside generated code.
            ///
            /// Generated from the table like everything else here, so an op
            /// cannot be added without one — which is the point. The
            /// hand-written match this replaced lived in
            /// `pixelflow-compiler`, covered 40 of the 50 ops, and closed
            /// with `_ => panic!("Unsupported OpKind for JIT")`.
            #[must_use]
            pub const fn variant_name(self) -> &'static str {
                match self { $( Self::$name => stringify!($name), )+ }
            }
        }
    };
}

op_table! {
    // --- Basic Arithmetic ---
    Var = 0,
    Const = 1,
    Add = 2,
    Sub = 3,
    Mul = 4,
    Div = 5,
    Neg = 6,
    Sqrt = 7,
    Rsqrt = 8,
    Abs = 9,
    Min = 10,
    Max = 11,
    MulAdd = 12,

    // --- Extended Math ---
    Recip = 13,
    Floor = 14,
    Ceil = 15,
    Round = 16,

    // --- Trigonometry ---
    Sin = 17,
    Cos = 18,
    Tan = 19,
    Asin = 20,
    Acos = 21,
    Atan = 22,
    Atan2 = 23,

    // --- Exponentials ---
    Exp = 24,
    Exp2 = 25,
    Ln = 26,
    Log2 = 27,
    Log10 = 28,
    Pow = 29,

    // --- Comparison ---
    Lt = 30,
    Le = 31,
    Gt = 32,
    Ge = 33,
    Eq = 34,
    Ne = 35,

    // --- Control Flow ---
    Select = 36,

    // --- Structure ---
    Tuple = 37,

    // --- Bit manipulation (integer-domain primitives) ---
    // These let exp/ln/log lower to arithmetic: each maps 1:1 to a single
    // hardware instruction. Lanes are reinterpreted as i32 where noted;
    // "reinterpret" itself is free (same register) and is not an op.
    /// Truncate f32 lanes to i32 (`cvttps2dq` / `fcvtzs`).
    TruncToInt = 38,
    /// Convert i32 lanes to f32 (`cvtdq2ps` / `scvtf`).
    IntToFloat = 39,
    /// Integer (i32) lane-wise add (`paddd` / `add .4s`).
    IAdd = 40,
    /// Logical shift-left of i32 lanes; RHS is a `Const` shift amount.
    Shl = 41,
    /// Logical shift-right of i32 lanes; RHS is a `Const` shift amount.
    Shr = 42,
    /// Bitwise AND of lane bit patterns (`andps` / `and`).
    BitAnd = 43,
    /// Bitwise OR of lane bit patterns (`orps` / `orr`).
    BitOr = 44,

    // --- Differentiation ---
    /// Symbolic derivative `∂(child0)/∂(var)` where `child1` is a `Const` whose
    /// value is the variable index (0=X, 1=Y, 2=Z, 3=W). The e-graph pushes
    /// `Dwrt` toward the leaves via the chain rule at optimization time;
    /// whatever survives (budget miss, unsupported op) falls to the runtime
    /// `lower_dwrt` pass, which either eliminates it or refuses to compile.
    /// It must never reach a backend.
    Dwrt = 45,

    // --- Bound memory (lattices) ---
    /// Buffer leaf: a slot referencing a `BufferDecl` in the arena's buffer
    /// table. The declared extents are static IR; the contents are bound at
    /// JIT-compile time. See `docs/designs/KERNELS_AND_LATTICES.md`.
    Buffer = 46,
    /// Read a bound buffer: `Gather(buffer, x, y)` where `buffer` is a
    /// `Buffer` leaf. Semantics match `DiscreteManifold::eval`: floor the
    /// indices, clamp to the declared extents, gather row-major.
    Gather = 47,
    /// Primitive gather: `RawGather(buffer, index)` reads `buffer`'s contents
    /// at the already-computed linear lane `index` (truncated to int), with no
    /// floor/clamp/row-major math. `Gather` lowers to index arithmetic (built
    /// from existing ops) plus this primitive — the analogue of `raw_mul` under
    /// `mul`. The index is trusted to be in bounds (the lowering clamps it).
    RawGather = 48,

    // --- Reduction (lattice fold) ---
    /// Fold a body over a bounded domain. Encoded
    /// `Nary(Reduce, [Const(combiner), Const(reduce_var), Const(extent), body])`:
    /// `combiner` is the monoid op index (`Add`/`Mul`/`Min`/`Max`), `body`
    /// references `Var(reduce_var)` (indices 4..8), and the fold runs over
    /// `0..extent`. The combiner is a *child* (a parameter), not baked into the
    /// opcode, so one `Reduce` covers every monoid and can later take an
    /// arbitrary combiner function. Lowered to an unrolled accumulation by
    /// `expand_reduce` before codegen — the analogue of `Gather -> RawGather`.
    Reduce = 49,

    // --- Uniforms (per-call scalars) ---
    /// Uniform leaf: a slot referencing a `UniformDecl` in the arena's
    /// uniform table. A scalar invariant across the lattice, supplied per
    /// call from a block; never folded, loaded once per call.
    Uniform = 50,

    /// An unbound scalar slot in a `kernel!` builder, before the builder is
    /// called. A leaf with no value yet — never folded, matched by no rewrite
    /// rule, derivative zero — which is exactly [`OpKind::Uniform`]'s
    /// contract, one tier earlier.
    ///
    /// Only the macro tier may hold one (`Vocabulary::Templates`); a `Param`
    /// reaching bake time means a builder was never called, and
    /// `Vocabulary::Runtime` declines it. Before this existed, the e-graph
    /// refused `Param` outright and every macro-side caller smuggled one past
    /// as something else — as `Var(16 + i)`, or as an opaque identifier —
    /// which is how `Var` reacquired the third meaning CLAUDE.md records as
    /// retired.
    Param = 51,
}

impl OpKind {
    /// Monoid identity for an op usable as a reduction combiner
    /// (`Add`→0, `Mul`→1, `Min`→+∞, `Max`→−∞). `None` if `self` is not a valid
    /// combiner.
    #[must_use]
    pub(crate) const fn monoid_identity(self) -> Option<f32> {
        match self {
            Self::Add => Some(0.0),
            Self::Mul => Some(1.0),
            Self::Min => Some(f32::INFINITY),
            Self::Max => Some(f32::NEG_INFINITY),
            // Mask monoids — the existential and universal quantifiers over a
            // bounded domain. Masks are bitwise in both evaluation tiers, so
            // the identities are the all-clear and all-set bit patterns
            // (`BitAnd`'s is a NaN by float reading; it is never read as a
            // number, only ANDed).
            Self::BitOr => Some(0.0),
            Self::BitAnd => Some(f32::from_bits(u32::MAX)),
            _ => None,
        }
    }

    /// Whether folding this op on `args` would bake in a **platform-specific**
    /// answer.
    ///
    /// The FP contract (CLAUDE.md, "Floating point at the edges") is hardware
    /// semantics — but the hardwares disagree in specific places, so for these
    /// (op, operand) combinations the language promises nothing:
    ///
    /// - `Min`/`Max` with a NaN operand: x86 `minps`/`maxps` compute
    ///   `(a OP b) ? a : b`, yielding the SECOND operand; aarch64 `FMIN`/`FMAX`
    ///   propagate the NaN. `min(NaN, 1.0)` is `1.0` on x86, `NaN` on aarch64.
    /// - `Min`/`Max` on **opposite-signed zeros**: the same operand-order rule
    ///   returns the second zero on x86 (since `-0.0 < 0.0` is false), while
    ///   `FMIN` selects `-0.0` and `FMAX` selects `+0.0`. `==` cannot see this
    ///   — `-0.0 == 0.0` — but `1.0 / x` turns it into `+inf` vs `-inf`.
    /// - `Gt`/`Ge` with a NaN operand: x86 uses the unordered predicates
    ///   (imm8 6/5), TRUE for NaN; aarch64 `FCMGT`/`FCMGE` are ordered, FALSE.
    /// - `Round` at an exact tie: nearest-even on x86 (`vroundps` imm 0x00),
    ///   ties-away on aarch64 (`FRINTA`), and `(x + 0.5).floor()` in the
    ///   combinator tier — three answers at `±n.5`.
    /// - `Round` on `-0.5 <= x <= -0.0`: both JIT tiers round to `-0.0`,
    ///   preserving the sign, while the combinator formula yields `+0.0`.
    ///   Not a tie, so it needs its own condition. `(x + 0.5).floor()` is not
    ///   any IEEE rounding mode — `round(-1.5)` is -1 there against -2
    ///   everywhere else — so the real repair is in that tier, not here.
    /// - `Recip`/`Rsqrt` **always**: these are estimate instructions and the
    ///   estimates differ — `rcpps` ~12 bits, `vrcp14ps` ~14, aarch64
    ///   `FRECPE` + one `FRECPS` refinement. The exact `1.0 / x` the folder
    ///   computes is not any of those answers.
    /// - `MulAdd` where one rounding differs from two: `vfmadd`/`FMLA` round
    ///   the product and sum together, SSE2's `mulps`+`addps` rounds twice.
    ///   `mul_add(1.0000001, 4097.0, 4097.0)` is `8194.001` fused, `8194.0`
    ///   split.
    ///
    /// x86 has no ties-away rounding mode and NaN/zero blending costs extra
    /// instructions, so unifying any of these would spend hot-path work on
    /// cases the language does not promise.
    ///
    /// **Why a fold in particular must decline:** constant folding happens on
    /// the BUILD host at macro-expansion time, but the constant it produces
    /// executes on the TARGET. Folding one of these bakes the build machine's
    /// arithmetic into a binary that may run somewhere computing differently —
    /// so this is not merely an optimized-vs-unoptimized discrepancy.
    ///
    /// `eval_unary`/`eval_binary` still return a value for these; it is this
    /// build's behavior and no promise about any other.
    #[must_use]
    pub fn fold_is_platform_specific(self, args: &[f32]) -> bool {
        match self {
            Self::Min | Self::Max => match args {
                // Equal but distinguishable is exactly ±0.0: x86 picks by
                // operand order, ARM by sign.
                [a, b] => a.is_nan() || b.is_nan() || (a == b && a.to_bits() != b.to_bits()),
                _ => false,
            },
            Self::Gt | Self::Ge => args.iter().any(|a| a.is_nan()),
            // Ties (x86 nearest-even vs aarch64 ties-away), plus the interval
            // where the combinator formula loses the sign of zero: for
            // `-0.5 <= x <= -0.0` both JIT tiers round to `-0.0` (sign
            // preserved) while `(x + 0.5).floor()` gives `+0.0`. `1.0 / x`
            // makes that `-inf` versus `+inf`. `-0.5` is already covered as a
            // tie; the rest of the interval is not.
            Self::Round => args.iter().any(|a| {
                (a - libm::truncf(*a)).abs() == 0.5 || (a.is_sign_negative() && *a > -0.5)
            }),
            // Reciprocal ESTIMATES, and every target estimates differently:
            // SSE2/AVX2 `rcpps`/`vrcpps` give ~12 bits, AVX-512 `vrcp14ps`
            // ~14, and aarch64 emits `FRECPE` plus one `FRECPS` Newton step.
            // `eval_unary`'s exact `1.0 / x` matches none of them, so there is
            // no host whose answer is worth baking. Unconditional: an estimate
            // is only guaranteed close, never equal, so no argument is safe.
            Self::Recip | Self::Rsqrt => true,
            // An invalid float->int conversion diverges in exactly two
            // places, and only two. x86 `cvttps2dq` yields the "integer
            // indefinite" 0x8000_0000 for every invalid input; aarch64
            // `FCVTZS` saturates to the destination's min/max with NaN going
            // to 0; and Rust's `as i32`, which `eval_unary` uses, saturates
            // like aarch64. So:
            //
            //   NaN        x86 -> i32::MIN,  aarch64 -> 0          DIVERGES
            //   x >= 2^31  x86 -> i32::MIN,  aarch64 -> i32::MAX   DIVERGES
            //   x < -2^31  x86 -> i32::MIN,  aarch64 -> i32::MIN   agrees
            //   -inf       x86 -> i32::MIN,  aarch64 -> i32::MIN   agrees
            //
            // The negative overflow direction agrees by arithmetic accident:
            // the integer-indefinite pattern IS i32::MIN, which is also what
            // saturation produces there. Refusing it too would cost ~19% of
            // the f32 space for nothing, so the predicate names the two real
            // cases rather than the convenient superset `!is_finite()`.
            //
            // The limit is 2^31 rather than `i32::MAX as f32`, because that
            // cast rounds UP to 2^31 and would admit values above the range.
            Self::TruncToInt => match args {
                [x] => {
                    const I32_LIMIT: f32 = 2_147_483_648.0; // 2^31
                    x.is_nan() || *x >= I32_LIMIT
                }
                _ => false,
            },

            // A shift count outside `0..32` has three answers. The folder
            // reduces it mod 32 (`y as u32 & 31`, below); x86
            // (V)PSLLD/(V)PSRLD zero the whole destination for any count > 31
            // (Intel SDM); and aarch64's immediate encoding silently changes
            // ELEMENT SIZE — `emit_shl` computes `immh:immb = shift + 32`, so
            // at shift >= 32 `immh` becomes `1xxx`, which decodes as `.2D`
            // and shifts bits across the 32-bit lane boundary. A non-integral
            // count is refused for the same reason: the tiers disagree on how
            // to narrow it. The emitters now assert this range, so a refused
            // fold cannot be laundered into a bad encoding either.
            Self::Shl | Self::Shr => match args {
                // The range test comes first and `||` short-circuits, so the
                // cast below only runs on a value already known to be in
                // `0..32` — where `as u32` is exact and total. (`f32::fract`
                // would be the obvious spelling but is std-only; this crate
                // is `no_std`.)
                [_, count] => !(0.0..32.0).contains(count) || (*count as u32) as f32 != *count,
                _ => false,
            },
            _ => false,
        }
    }

    /// Whether `self` is a valid reduction combiner (an associative monoid op
    /// with an identity).
    #[must_use]
    pub(crate) const fn is_monoid(self) -> bool {
        self.monoid_identity().is_some()
    }

    /// Every op, in [`OpKind::index`] order.
    ///
    /// Prefer this to walking `0..COUNT` and calling [`OpKind::from_index`] —
    /// that spelling is why the three ops sitting past the old discriminant
    /// gaps were never visited by any table-filling loop.
    pub fn all() -> impl Iterator<Item = Self> + Clone {
        Self::ALL.into_iter()
    }

    /// Get the arity of the operation.
    #[must_use]
    pub const fn arity(self) -> usize {
        match self {
            Self::Var | Self::Const | Self::Tuple | Self::Buffer | Self::Uniform | Self::Param => 0,

            Self::Neg
            | Self::Sqrt
            | Self::Rsqrt
            | Self::Abs
            | Self::Recip
            | Self::Floor
            | Self::Ceil
            | Self::Round
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Asin
            | Self::Acos
            | Self::Atan
            | Self::Exp
            | Self::Exp2
            | Self::Ln
            | Self::Log2
            | Self::Log10
            | Self::TruncToInt
            | Self::IntToFloat => 1,

            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Min
            | Self::Max
            | Self::Atan2
            | Self::Pow
            | Self::Lt
            | Self::Le
            | Self::Gt
            | Self::Ge
            | Self::Eq
            | Self::Ne
            | Self::IAdd
            | Self::Shl
            | Self::Shr
            | Self::BitAnd
            | Self::BitOr
            | Self::Dwrt
            | Self::RawGather => 2,

            Self::MulAdd | Self::Select | Self::Gather => 3,

            // The body. The algebra, the binder and the range are the node's
            // own metadata ([`Fold`](crate::Fold)) rather than three `Const`
            // children, so nothing that walks operands can reach them.
            Self::Reduce => 1,
        }
    }

    /// Get the display name of the operation.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Var => "var",
            Self::Const => "const",
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Neg => "neg",
            Self::Sqrt => "sqrt",
            Self::Rsqrt => "rsqrt",
            Self::Abs => "abs",
            Self::Min => "min",
            Self::Max => "max",
            Self::MulAdd => "mul_add",
            Self::Recip => "recip",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Round => "round",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Atan2 => "atan2",
            Self::Exp => "exp",
            Self::Exp2 => "exp2",
            Self::Ln => "ln",
            Self::Log2 => "log2",
            Self::Log10 => "log10",
            Self::Pow => "pow",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Select => "select",
            Self::Tuple => "tuple",
            Self::TruncToInt => "trunc_to_int",
            Self::IntToFloat => "int_to_float",
            Self::IAdd => "iadd",
            Self::Shl => "shl",
            Self::Shr => "shr",
            Self::BitAnd => "bitand",
            Self::BitOr => "bitor",
            Self::Dwrt => "dwrt",
            Self::Buffer => "buffer",
            Self::Gather => "gather",
            Self::RawGather => "raw_gather",
            Self::Reduce => "reduce",
            Self::Uniform => "uniform",
            Self::Param => "param",
        }
    }

    /// Parse OpKind from its string name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "var" => Some(Self::Var),
            "const" => Some(Self::Const),
            "add" => Some(Self::Add),
            "sub" => Some(Self::Sub),
            "mul" => Some(Self::Mul),
            "div" => Some(Self::Div),
            "neg" => Some(Self::Neg),
            "sqrt" => Some(Self::Sqrt),
            "rsqrt" => Some(Self::Rsqrt),
            "abs" => Some(Self::Abs),
            "min" => Some(Self::Min),
            "max" => Some(Self::Max),
            "mul_add" => Some(Self::MulAdd),
            "recip" => Some(Self::Recip),
            "floor" => Some(Self::Floor),
            "ceil" => Some(Self::Ceil),
            "round" => Some(Self::Round),
            "sin" => Some(Self::Sin),
            "cos" => Some(Self::Cos),
            "tan" => Some(Self::Tan),
            "asin" => Some(Self::Asin),
            "acos" => Some(Self::Acos),
            "atan" => Some(Self::Atan),
            "atan2" => Some(Self::Atan2),
            "exp" => Some(Self::Exp),
            "exp2" => Some(Self::Exp2),
            "ln" => Some(Self::Ln),
            "log2" => Some(Self::Log2),
            "log10" => Some(Self::Log10),
            "pow" | "powf" => Some(Self::Pow),
            "lt" => Some(Self::Lt),
            "le" => Some(Self::Le),
            "gt" => Some(Self::Gt),
            "ge" => Some(Self::Ge),
            "eq" => Some(Self::Eq),
            "ne" => Some(Self::Ne),
            "select" => Some(Self::Select),
            "tuple" => Some(Self::Tuple),
            "trunc_to_int" => Some(Self::TruncToInt),
            "int_to_float" => Some(Self::IntToFloat),
            "iadd" => Some(Self::IAdd),
            "shl" => Some(Self::Shl),
            "shr" => Some(Self::Shr),
            "bitand" => Some(Self::BitAnd),
            "bitor" => Some(Self::BitOr),
            "dwrt" => Some(Self::Dwrt),
            "buffer" => Some(Self::Buffer),
            "gather" => Some(Self::Gather),
            "raw_gather" => Some(Self::RawGather),
            "reduce" => Some(Self::Reduce),
            "uniform" => Some(Self::Uniform),
            "param" => Some(Self::Param),
            _ => None,
        }
    }

    /// True for ops a kernel body may call as `.method(args)` — the surface
    /// set [`OpKind::from_method_call`] resolves into.
    ///
    /// Stricter than "not [`EmitStyle::Special`]": [`Self::Add`]/[`Self::Sub`]/
    /// [`Self::Mul`]/[`Self::Div`] are real, non-special ops but are spelled
    /// `+ - * /`, never `.add(y)`, and [`Self::TruncToInt`]/[`Self::IAdd`]/
    /// [`Self::Shl`]/[`Self::Shr`]/[`Self::BitAnd`]/[`Self::BitOr`]/
    /// [`Self::IntToFloat`] are primitives that only ever arise from lowering
    /// passes (`Gather`/`Reduce` expansion), never from surface syntax a
    /// kernel body writes directly.
    #[must_use]
    const fn is_dsl_method(self) -> bool {
        matches!(
            self,
            Self::Neg
                | Self::Sqrt
                | Self::Rsqrt
                | Self::Abs
                | Self::Recip
                | Self::Floor
                | Self::Ceil
                | Self::Round
                | Self::Sin
                | Self::Cos
                | Self::Tan
                | Self::Asin
                | Self::Acos
                | Self::Atan
                | Self::Exp
                | Self::Exp2
                | Self::Ln
                | Self::Log2
                | Self::Log10
                | Self::Min
                | Self::Max
                | Self::Atan2
                | Self::Pow
                | Self::Lt
                | Self::Le
                | Self::Gt
                | Self::Ge
                | Self::Eq
                | Self::Ne
                | Self::MulAdd
                | Self::Select
        )
    }

    /// Resolve a DSL method call — as written in a `kernel!` body, e.g.
    /// `.sqrt()` or `.min(y)` — to the primitive op it denotes.
    ///
    /// `arg_count` is the call's argument list length, not counting the
    /// receiver; this checks it against the op's arity, so `"sqrt"` at
    /// `arg_count` 1 is rejected exactly as an unrecognized name would be.
    /// `None` for anything [`OpKind::is_dsl_method`] excludes, or a name
    /// [`OpKind::from_name`] doesn't recognize at all.
    ///
    /// This is the one place `(name, arity) -> op` is decided; a compiler
    /// front end that re-lists method names itself is a second place for that
    /// mapping to drift from this one.
    #[must_use]
    pub fn from_method_call(name: &str, arg_count: usize) -> Option<Self> {
        let op = Self::from_name(name)?;
        if !op.is_dsl_method() {
            return None;
        }
        (op.arity() == arg_count + 1).then_some(op)
    }

    /// Get the default cost estimate for this operation (in cycles).
    #[must_use]
    pub const fn default_cost(self) -> usize {
        match self {
            Self::Var | Self::Const | Self::Tuple | Self::Buffer | Self::Uniform | Self::Param => 0,
            // Memory read: native gather on AVX2/AVX-512, scalar loads on
            // NEON/SSE2. Priced between an arithmetic op and a transcendental.
            Self::Gather | Self::RawGather => 10,
            // Reduction is lowered (unrolled) away before costing; price the
            // node itself at zero so a stray one never dominates extraction.
            Self::Reduce => 0,
            Self::Neg | Self::Abs | Self::Floor | Self::Ceil | Self::Round => 1,
            Self::Add
            | Self::Sub
            | Self::Min
            | Self::Max
            | Self::Lt
            | Self::Le
            | Self::Gt
            | Self::Ge
            | Self::Eq
            | Self::Ne
            | Self::Select => 4,
            Self::Mul | Self::MulAdd | Self::Recip | Self::Rsqrt => 5,
            // Bit-manip primitives: single cheap integer/convert instructions.
            Self::TruncToInt
            | Self::IntToFloat
            | Self::IAdd
            | Self::Shl
            | Self::Shr
            | Self::BitAnd
            | Self::BitOr => 1,
            // Dwrt must be rewritten away before extraction; price it so the
            // extractor never prefers a surviving derivative over a decomposed
            // form. (A surviving Dwrt is then caught by a validation pass.)
            Self::Dwrt => 1_000_000,
            Self::Div
            | Self::Sqrt
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Asin
            | Self::Acos
            | Self::Atan
            | Self::Atan2
            | Self::Exp
            | Self::Exp2
            | Self::Ln
            | Self::Log2
            | Self::Log10
            | Self::Pow => 15,
        }
    }

    /// Returns true if the operation is commutative (a op b == b op a).
    #[must_use]
    pub const fn is_commutative(self) -> bool {
        matches!(
            self,
            Self::Add | Self::Mul | Self::Min | Self::Max | Self::Eq | Self::Ne
        )
    }

    /// Returns true if the operation is associative ((a op b) op c == a op (b op c)).
    #[must_use]
    pub const fn is_associative(self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::Min | Self::Max)
    }

    /// Returns the identity element if one exists (a op identity == a).
    #[must_use]
    pub const fn identity(self) -> Option<f32> {
        match self {
            Self::Add | Self::Sub => Some(0.0),
            Self::Mul | Self::Div => Some(1.0),
            _ => None,
        }
    }

    /// Returns the annihilator element if one exists (a op annihilator == annihilator).
    #[must_use]
    pub const fn annihilator(self) -> Option<f32> {
        match self {
            Self::Mul => Some(0.0),
            _ => None,
        }
    }

    /// Returns true if the operation is idempotent (a op a == a).
    #[must_use]
    pub const fn is_idempotent(self) -> bool {
        matches!(self, Self::Min | Self::Max | Self::Abs)
    }

    /// Returns true if this op should appear in randomly generated seed expressions
    /// fed to the JIT training pipeline.
    ///
    /// Excludes:
    /// - Var/Const (leaves, not ops)
    /// - Tuple (structural, not computational)
    /// - MulAdd (fused — should only arise from rewrite rules)
    /// - Lt/Le/Gt/Ge/Eq/Ne (return masks, not floats — type-invalid in arithmetic)
    /// - Select (needs mask input — only valid composed with a comparison)
    /// - Buffer/Gather (memory ops — require a bound buffer, not synthesizable)
    /// - Uniform/Param (unbound scalar slots — a value arrives per call or
    ///   from a builder, so a generator cannot synthesize one that evaluates)
    #[must_use]
    pub const fn is_seed_op(self) -> bool {
        !matches!(
            self,
            Self::Var
                | Self::Const
                | Self::Tuple
                | Self::MulAdd
                | Self::Lt
                | Self::Le
                | Self::Gt
                | Self::Ge
                | Self::Eq
                | Self::Ne
                | Self::Select
                | Self::Buffer
                | Self::Gather
                | Self::RawGather
                | Self::Reduce
                | Self::Uniform
                | Self::Param
        )
    }

    /// Get the emit style for code generation.
    #[must_use]
    pub const fn emit_style(self) -> EmitStyle {
        match self {
            // Special cases handled separately
            Self::Var | Self::Const | Self::Tuple | Self::Param => EmitStyle::Special,

            // Unary prefix: (-a)
            Self::Neg => EmitStyle::UnaryPrefix,

            // Unary method: (a).sqrt()
            Self::Sqrt
            | Self::Rsqrt
            | Self::Abs
            | Self::Recip
            | Self::Floor
            | Self::Ceil
            | Self::Round
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Asin
            | Self::Acos
            | Self::Atan
            | Self::Exp
            | Self::Exp2
            | Self::Ln
            | Self::Log2
            | Self::Log10
            | Self::TruncToInt
            | Self::IntToFloat => EmitStyle::UnaryMethod,

            // Binary infix: (a + b)
            Self::Add => EmitStyle::BinaryInfix("+"),
            Self::Sub => EmitStyle::BinaryInfix("-"),
            Self::Mul => EmitStyle::BinaryInfix("*"),
            Self::Div => EmitStyle::BinaryInfix("/"),

            // Binary method: (a).min(b)
            Self::Min
            | Self::Max
            | Self::Atan2
            | Self::Pow
            | Self::Lt
            | Self::Le
            | Self::Gt
            | Self::Ge
            | Self::Eq
            | Self::Ne
            | Self::IAdd
            | Self::Shl
            | Self::Shr
            | Self::BitAnd
            | Self::BitOr => EmitStyle::BinaryMethod,

            // Differentiation: never emitted (rewritten away in the e-graph).
            Self::Dwrt => EmitStyle::Special,

            // Memory ops and uniforms: emitted by the JIT binding path, not
            // as method calls.
            Self::Buffer | Self::Gather | Self::RawGather | Self::Uniform => EmitStyle::Special,

            // Reduction: lowered to unrolled arithmetic before codegen.
            Self::Reduce => EmitStyle::Special,

            // Ternary method: (a).mul_add(b, c)
            Self::MulAdd | Self::Select => EmitStyle::TernaryMethod,
        }
    }

    /// Evaluate a unary operation on a constant argument.
    ///
    /// Returns `None` for non-unary operations or operations that can't be
    /// evaluated at compile time.
    #[must_use]
    pub fn eval_unary(self, x: f32) -> Option<f32> {
        match self {
            Self::Neg => Some(-x),
            Self::Sqrt => Some(libm::sqrtf(x)),
            Self::Rsqrt => Some(1.0 / libm::sqrtf(x)),
            Self::Abs => Some(x.abs()),
            Self::Recip => Some(1.0 / x),
            Self::Floor => Some(libm::floorf(x)),
            Self::Ceil => Some(libm::ceilf(x)),
            // x86's answer (`vroundps` imm 0x00 is nearest-even). aarch64
            // `FRINTA` is ties-away and the combinator tier is
            // `(x + 0.5).floor()`, so at a tie this is one of three behaviors
            // and no promise — see `fold_is_platform_specific`.
            Self::Round => Some(libm::rintf(x)),
            // Integer-domain primitives. A lane holds an f32 bit pattern; these
            // reinterpret it as i32, exactly as the hardware instructions do
            // (`cvttps2dq` / `cvtdq2ps`). They exist so exp/log can lower to
            // arithmetic, and the interpreter needs them for the same reason —
            // it evaluates that lowering.
            Self::TruncToInt => Some(f32::from_bits((x as i32) as u32)),
            Self::IntToFloat => Some((x.to_bits() as i32) as f32),

            // Transcendentals have no arm on purpose. `f32::sin` and friends
            // are std-only (absent from `core`) and would be a *second*
            // definition of a function the compiler already defines by
            // expansion. Callers lower first — `expand_transcendentals` — and
            // then only the primitives above are ever evaluated. Same
            // discipline as `Dwrt`, which is also lowered rather than
            // interpreted.
            _ => None,
        }
    }

    /// A comparison lane: all bits set for true, all clear for false.
    ///
    /// This is not a stylistic choice about how to spell a boolean — it is the
    /// only representation the consumers accept. `Select` is a *bitwise* blend
    /// on every backend (`andps`/`andnps`/`orps` on SSE2 and AVX2, one
    /// `vpternlogd 0xCA` on AVX-512, `BSL` on aarch64), and `BitAnd`/`BitOr`
    /// are literal bitwise ops, so a mask lane's job is to be a per-bit
    /// stencil. `1.0` is not one: `0xFFFFFFFF & 0x3f800000` is `0x3f800000`,
    /// and blending `7.0` against `9.0` through that pattern gives `4.5` — a
    /// value neither branch ever held.
    ///
    /// Every tier that *executes* a comparison already writes all-ones —
    /// `cmpps`, `FCMEQ`, and the combinator tier's SIMD compares alike. Folding
    /// in the same representation is what makes a folded comparison and an
    /// executed one interchangeable as operands.
    #[must_use]
    pub const fn mask(condition: bool) -> f32 {
        if condition {
            f32::from_bits(u32::MAX)
        } else {
            0.0
        }
    }

    /// Whether this op's result is a lane **bit pattern** rather than a number.
    ///
    /// Comparisons produce masks ([`Self::mask`] — all-ones, which reads as
    /// NaN); the integer-domain primitives reinterpret lanes as `i32` and are
    /// how exp/log lower to arithmetic; `Select`/`BitAnd`/`BitOr` pass patterns
    /// through. For all of them a non-finite float reading is the intended
    /// output rather than an overflow, which is what separates them from
    /// `1e38 * 1e38` — the case a folder is right to refuse.
    #[must_use]
    pub const fn is_bitwise_domain(self) -> bool {
        matches!(
            self,
            Self::Lt
                | Self::Le
                | Self::Gt
                | Self::Ge
                | Self::Eq
                | Self::Ne
                | Self::Select
                | Self::BitAnd
                | Self::BitOr
                | Self::TruncToInt
                | Self::IntToFloat
                | Self::IAdd
                | Self::Shl
                | Self::Shr
        )
    }

    /// Evaluate a binary operation on constant arguments.
    ///
    /// Returns `None` for non-binary operations.
    #[must_use]
    pub fn eval_binary(self, x: f32, y: f32) -> Option<f32> {
        match self {
            Self::Add => Some(x + y),
            Self::Sub => Some(x - y),
            Self::Mul => Some(x * y),
            Self::Div => Some(x / y),
            // `minps`/`maxps` semantics on x86: `(a OP b) ? a : b`, so a NaN in
            // EITHER operand yields the second. aarch64 `FMIN`/`FMAX` propagate
            // the NaN instead — see `fold_is_platform_specific`. This arm
            // is x86's answer and is not a promise for other targets.
            Self::Min => Some(if x < y { x } else { y }),
            Self::Max => Some(if x > y { x } else { y }),
            Self::Lt => Some(Self::mask(x < y)),
            Self::Le => Some(Self::mask(x <= y)),
            // Gt/Ge are x86's UNORDERED predicates (imm8 6 = NLE_US, 5 =
            // NLT_US), so unlike Lt/Le they are TRUE when either operand is
            // NaN. ARM's FCMGT/FCMGE are ordered and give FALSE — see
            // `fold_is_platform_specific`.
            // Spelled as the ordered comparison plus the unordered case rather
            // than as `!(x <= y)`: same answer, but it says which part is the
            // NaN behavior instead of hiding it inside a negation.
            Self::Gt => Some(Self::mask(x > y || x.is_nan() || y.is_nan())),
            Self::Ge => Some(Self::mask(x >= y || x.is_nan() || y.is_nan())),
            // Exact, as every emitter compares. The old `(x-y).abs() < EPSILON`
            // made the folder disagree with the JIT for ordinary finite values:
            // 0.5 and its next representable neighbour differ by less than
            // EPSILON, so folding said "equal" where the hardware says "not".
            // NaN is false here and in `cmpps`/`FCMEQ` alike.
            Self::Eq => Some(Self::mask(x == y)),
            // Exact, and true for NaN — matching `NEQ_UQ` and an inverted
            // `FCMEQ`. Same epsilon bug as `Eq` above.
            Self::Ne => Some(Self::mask(x != y)),
            // Integer-domain binaries on lane bit patterns. The shift count is
            // the RHS `Const`'s numeric value (the schedule reads it as
            // `*v as u32 as u8`), and both shifts are logical — `vpslld` /
            // `vpsrld`, zero-filling.
            Self::IAdd => Some(f32::from_bits(
                (x.to_bits() as i32).wrapping_add(y.to_bits() as i32) as u32,
            )),
            Self::Shl => Some(f32::from_bits(x.to_bits() << (y as u32 & 31))),
            Self::Shr => Some(f32::from_bits(x.to_bits() >> (y as u32 & 31))),
            // Bitwise on lane bit patterns, matching the SIMD backends. On
            // canonical masks (1.0/0.0 here, all-ones/zero in the JIT) this
            // is logical AND/OR in both representations.
            Self::BitAnd => Some(f32::from_bits(x.to_bits() & y.to_bits())),
            Self::BitOr => Some(f32::from_bits(x.to_bits() | y.to_bits())),
            _ => None,
        }
    }

    /// Evaluate a ternary operation on constant arguments.
    #[must_use]
    pub fn eval_ternary(self, x: f32, y: f32, z: f32) -> Option<f32> {
        match self {
            // One rounding, always. `MulAdd` denotes `x*y + z`; whether a
            // target spells it as one instruction (`vfmadd`, `FMLA`) or as a
            // multiply and an add (SSE2, or any backend under register
            // pressure — see `ResolvedOp::DecomposedMulAdd`) is a last-bit
            // precision difference inside the contract, not a divergence, so
            // it folds unconditionally. `libm::fmaf` rather than `x * y + z`
            // because the latter is whatever the compiler that built THIS
            // crate chose to contract it into: under `-fp-contract=fast` a
            // fold's answer must not depend on the folder's build profile.
            Self::MulAdd => Some(libm::fmaf(x, y, z)),
            // The bitwise blend every backend emits — `andps`/`andnps`/`orps`
            // on SSE2 and AVX2, one `vpternlogd 0xCA` on AVX-512, `BSL` on
            // aarch64. Spelling it `if x != 0.0` would be right only for a
            // canonical [`Self::mask`] and silently wrong for anything else.
            Self::Select => Some(f32::from_bits(
                (x.to_bits() & y.to_bits()) | (!x.to_bits() & z.to_bits()),
            )),
            _ => None,
        }
    }
}

// `op_table!` makes the enum, the roster, the count and both directions agree
// by generating them from one list — but it takes the numbers on faith. It
// cannot tell whether the numbers written there are dense.
//
// That is what this checks. Give an entry a number that skips one (`Const = 5`
// after `Var = 0`) and every generated item is still internally consistent,
// while `ALL[1]` no longer sits at index 1 — a per-op table would then have a
// slot nothing reaches, and the ops past the gap would run off its end. Which
// is the original bug, and the reason `COUNT` counts entries rather than
// naming the largest number.
//
// Duplicates need no help here: two entries sharing a number is E0081 on the
// discriminants before this block ever runs.
const _: () = {
    let mut i = 0;
    while i < OpKind::COUNT {
        let op = OpKind::ALL[i];
        assert!(op.index() == i, "OpKind::ALL is out of index() order");
        assert!(op as usize == i, "discriminant disagrees with index()");
        match OpKind::from_index(i) {
            Some(back) => assert!(back.index() == i, "from_index is not index()'s inverse"),
            None => panic!("from_index has a gap inside 0..COUNT"),
        }
        i += 1;
    }
    assert!(
        OpKind::from_index(OpKind::COUNT).is_none(),
        "from_index answers past COUNT"
    );
};

// ============================================================================
// Marshalling
// ============================================================================

/// An [`OpKind`] in transit — the form it takes in a byte stream.
///
/// **The bytes are this crate's business, not yours.** What is promised is the
/// round trip: [`OpKind::unmarshal`] undoes [`OpKind::marshal`] for every op,
/// and [`OpCode::SIZE`] is however wide that happens to be right now. What is
/// *not* promised is which value any op encodes to, that the value is stable
/// across releases, or that `SIZE` stays 1. Persisted data must therefore
/// carry a format version and refuse anything it does not recognise, the way
/// the training corpus does — a stale file has to fail loudly rather than
/// decode into the wrong ops.
///
/// Handing out the table subscript instead would leak: a consumer that writes
/// `index()` into its own format has made this crate's private numbering part
/// of that format, and owes it a version bump for a change it cannot see.
/// See `docs/designs/opkind-numbering-is-private.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpCode(u8);

impl OpCode {
    /// Width of an encoded op, in bytes.
    ///
    /// A `const`, not a literal, so that widening the encoding is a
    /// recompile at every call site rather than a silent truncation.
    pub const SIZE: usize = 1;

    /// The encoded bytes, to hand to a writer.
    #[must_use]
    pub const fn to_bytes(self) -> [u8; Self::SIZE] {
        [self.0]
    }

    /// Rebuild from bytes previously produced by [`OpCode::to_bytes`].
    ///
    /// Any byte is accepted here; whether it names an op is
    /// [`OpKind::unmarshal`]'s answer.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; Self::SIZE]) -> Self {
        Self(bytes[0])
    }
}

impl OpKind {
    /// Encode for transmission or storage. See [`OpCode`].
    #[must_use]
    pub const fn marshal(self) -> OpCode {
        // The subscript is a convenient encoding today. Nothing outside may
        // depend on that staying true, which is the whole point of `OpCode`.
        OpCode(self.index() as u8)
    }

    /// Decode. `None` if the code names no op — which is what a truncated,
    /// corrupt, or stale-format stream looks like from in here.
    #[must_use]
    pub const fn unmarshal(code: OpCode) -> Option<Self> {
        Self::from_index(code.0 as usize)
    }
}

// `marshal` narrows the subscript to one byte. That is sound only while the
// op set fits in a byte, and it would otherwise wrap silently — so it is
// checked here rather than left to whoever adds the 257th op.
const _: () = assert!(
    OpKind::COUNT <= u8::MAX as usize + 1,
    "op set outgrew OpCode's width: widen OpCode::SIZE"
);

// ============================================================================
// OpMap
// ============================================================================

/// A total map from every [`OpKind`] to a `T`.
///
/// This exists because the obvious alternative — `[T; OpKind::COUNT]`
/// subscripted by `OpKind::index()` — spread the same defect to every consumer
/// that reached for it. Four of them did: the latency-prior cost table, the
/// learned cost model, the NNUE op embeddings, and its gradient buffer.
///
/// Two things go wrong with the bare array, and both went wrong here:
///
/// 1. **The subscript escapes.** `index() -> usize` hands out an integer with
///    no memory of the bound it is valid against, so each consumer re-derives
///    the same unchecked cast. When the discriminants had gaps, every one of
///    them was out of bounds for the last three ops — and the workaround that
///    appeared was a local `if op.index() < OpKind::COUNT` guard that
///    *silently skipped* those ops rather than failing.
/// 2. **Positional literals drift.** A `[T; COUNT]` written out in order is
///    aligned to the discriminants only by convention, and nothing checks the
///    convention. The latency prior was written densely while `index()` was
///    sparse, so 13 of 50 ops read a neighbour's cost — `Dwrt` was priced at
///    10 instead of the 1000 that keeps extraction away from it.
///
/// [`OpMap::from_fn`] closes the second hole: an exhaustive `match` on
/// `OpKind` is checked by the compiler, so a table cannot silently misalign
/// and cannot silently omit an op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpMap<T> {
    slots: [T; OpKind::COUNT],
}

impl<T> OpMap<T> {
    /// Number of slots — one per op.
    ///
    /// This is the public spelling of "how many ops are there", for sizing
    /// things built alongside a per-op table. It says nothing about which
    /// number any particular op sits at.
    pub const LEN: usize = OpKind::COUNT;

    /// Build a table by answering for every op. Write the body as a `match`
    /// and the compiler will not let you forget one.
    pub fn from_fn(mut f: impl FnMut(OpKind) -> T) -> Self {
        Self {
            slots: core::array::from_fn(|i| f(OpKind::ALL[i])),
        }
    }

    /// Iterate in [`OpKind::index`] order — the order [`OpMap::as_slice`]
    /// serializes in.
    pub fn iter(&self) -> impl Iterator<Item = (OpKind, &T)> {
        OpKind::ALL.into_iter().zip(self.slots.iter())
    }

    /// The backing slots in `index()` order. For serialization; prefer
    /// indexing by [`OpKind`] for lookups.
    pub fn as_slice(&self) -> &[T] {
        &self.slots
    }

    /// Mutable backing slots in `index()` order. For deserialization.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.slots
    }
}

impl<T: Copy> OpMap<T> {
    /// A table where every op answers `value`.
    pub const fn splat(value: T) -> Self {
        Self {
            slots: [value; OpKind::COUNT],
        }
    }
}

impl<T: Default + Copy> Default for OpMap<T> {
    fn default() -> Self {
        Self::splat(T::default())
    }
}

impl<T> Index<OpKind> for OpMap<T> {
    type Output = T;

    #[inline]
    fn index(&self, op: OpKind) -> &T {
        // The only place in the workspace where an op becomes a subscript.
        &self.slots[op.index()]
    }
}

impl<T> IndexMut<OpKind> for OpMap<T> {
    #[inline]
    fn index_mut(&mut self, op: OpKind) -> &mut T {
        &mut self.slots[op.index()]
    }
}

// EmitStyle is imported from crate::traits - single source of truth

/// Every op name the surface language accepts as a method — see
/// [`OpKind::is_dsl_method`] for exactly which ops that is and why it is
/// narrower than "not `Special`".
///
/// This used to walk a parallel array of one zero-sized type per op, each
/// implementing an `OpMeta`/`Op`/arity-marker trait family, purely to answer
/// this question. `OpKind` already knows every op's name and emit style, so
/// the family and its 230-line module are gone.
pub fn known_method_names() -> impl Iterator<Item = &'static str> {
    OpKind::all()
        .filter(|op| op.is_dsl_method())
        .map(OpKind::name)
        .chain(METHOD_ALIASES.iter().map(|(spelling, _)| *spelling))
}

/// Accepted spellings that are not their op's canonical [`OpKind::name`].
///
/// `from_name` takes `"powf"` because `f32::powf` is what a Rust programmer
/// types; the canonical name is `"pow"`. Both resolve, so both must be
/// advertised — a spelling the front end accepts but never lists is one no
/// sweep exercises and no consumer can discover, which is exactly how a later
/// stage could drop `powf` with every test still green.
///
/// The aliases live here, beside [`known_method_names`], rather than only
/// inside `from_name`'s match: accepting a name and advertising it are the
/// same fact, and this is the one place that says so.
const METHOD_ALIASES: &[(&str, OpKind)] = &[("powf", OpKind::Pow)];

#[cfg(test)]
mod index_space {
    use super::{OpCode, OpKind};

    /// Density, totality and round-tripping are proved at compile time by the
    /// `const` block above, so there is no runtime test for them here.
    ///
    /// There is deliberately no test pinning op N to a particular number.
    /// The numbering is this crate's own business — see [`OpCode`] — and a
    /// test asserting `Add == 2` would turn an internal detail into a promise
    /// that consumers can quietly build on, leaving this crate owing a version
    /// bump to every format that took it up. Persisted data is protected by
    /// its own format version, not by freezing these values.
    ///
    /// What the round trip must do is survive the trip.
    #[test]
    fn every_op_survives_marshalling() {
        for op in OpKind::all() {
            let bytes = op.marshal().to_bytes();
            let back = OpKind::unmarshal(OpCode::from_bytes(bytes));
            assert_eq!(back, Some(op), "{op:?} did not survive marshalling");
        }
    }

    /// A code naming no op is a `None`, not a wrong op and not a panic — this
    /// is the untrusted-input path (a truncated or stale stream).
    #[test]
    fn codes_naming_no_op_decode_to_none() {
        let live: alloc::vec::Vec<[u8; OpCode::SIZE]> =
            OpKind::all().map(|op| op.marshal().to_bytes()).collect();
        for b in 0..=u8::MAX {
            let decoded = OpKind::unmarshal(OpCode::from_bytes([b]));
            assert_eq!(
                decoded.is_some(),
                live.contains(&[b]),
                "byte {b} decoded to {decoded:?}"
            );
        }
    }

    /// `from_name` matches on `&str` and can never be exhaustive, so nothing
    /// makes a new op's name mandatory. Written name-first because the reverse
    /// is not injective: `"pow"` and `"powf"` both parse to `Pow`.
    #[test]
    fn every_op_round_trips_through_its_name() {
        for op in OpKind::all() {
            assert_eq!(
                OpKind::from_name(op.name()),
                Some(op),
                "{op:?} names {:?}, which does not parse back",
                op.name()
            );
        }
    }
}

#[cfg(test)]
mod op_map {
    use super::{OpKind, OpMap};

    #[test]
    fn iter_yields_every_op_paired_with_its_own_slot_in_index_order() {
        let map = OpMap::from_fn(OpKind::index);
        let collected: Vec<(OpKind, usize)> = map.iter().map(|(op, v)| (op, *v)).collect();

        assert_eq!(collected.len(), OpKind::COUNT);
        for (i, (op, value)) in collected.iter().enumerate() {
            assert_eq!(op.index(), i, "iter() is not in index() order at slot {i}");
            assert_eq!(*value, i, "slot {i} does not hold its own op's value");
        }
    }

    #[test]
    fn as_slice_exposes_the_same_values_iter_yields_in_the_same_order() {
        let map = OpMap::from_fn(|op| op.index());

        let via_slice: Vec<usize> = map.as_slice().to_vec();
        let via_iter: Vec<usize> = map.iter().map(|(_, v)| *v).collect();

        assert_eq!(via_slice, via_iter);
    }

    #[test]
    fn as_mut_slice_writes_are_visible_through_indexing_by_op() {
        let mut map: OpMap<usize> = OpMap::splat(0);
        map.as_mut_slice()[OpKind::Add.index()] = 99;

        assert_eq!(map[OpKind::Add], 99, "write through as_mut_slice was lost");
        assert_eq!(
            map[OpKind::Sub],
            0,
            "as_mut_slice write bled into a neighboring op's slot"
        );
    }

    #[test]
    fn default_matches_splat_of_the_value_type_default() {
        let defaulted: OpMap<usize> = OpMap::default();
        let splatted: OpMap<usize> = OpMap::splat(usize::default());

        assert_eq!(defaulted, splatted);
    }
}

#[cfg(test)]
mod method_names {
    use super::{EmitStyle, OpKind, known_method_names};

    /// Every advertised name must resolve. Not every advertised name is its
    /// own canonical spelling — `from_name` is deliberately non-injective, and
    /// `"powf"` resolves to `Pow`, whose `name()` is `"pow"`. This asserted
    /// the stronger property while aliases were unadvertised, which is a fine
    /// thing to have asserted and the wrong thing to keep asserting: it would
    /// now forbid advertising exactly the spellings that most need it.
    #[test]
    fn every_returned_name_resolves() {
        for name in known_method_names() {
            assert!(
                OpKind::from_name(name).is_some(),
                "known_method_names() produced {name:?}, which from_name() cannot parse"
            );
        }
    }

    /// Each alias resolves to the op it claims, and is genuinely an alias
    /// rather than a name that has quietly become canonical.
    #[test]
    fn every_alias_resolves_to_its_op_under_a_different_canonical_name() {
        for (spelling, op) in super::METHOD_ALIASES {
            assert_eq!(OpKind::from_name(spelling), Some(*op));
            assert_ne!(
                *spelling,
                op.name(),
                "{spelling:?} is the canonical name, so it does not belong in METHOD_ALIASES"
            );
        }
    }

    #[test]
    fn excludes_every_op_whose_emit_style_is_special() {
        let names: std::collections::HashSet<&str> = known_method_names().collect();

        for op in OpKind::all() {
            if matches!(op.emit_style(), EmitStyle::Special) {
                assert!(
                    !names.contains(op.name()),
                    "{op:?} has EmitStyle::Special and must not have a method spelling"
                );
            }
        }
    }

    #[test]
    fn includes_an_ordinary_binary_op_by_its_method_name() {
        let names: Vec<&str> = known_method_names().collect();
        assert!(
            names.contains(&"min"),
            "Min is not EmitStyle::Special and should surface as a known method"
        );
    }

    #[test]
    fn excludes_infix_arithmetic_spelled_as_operators_not_methods() {
        let names: Vec<&str> = known_method_names().collect();
        for op in ["add", "sub", "mul", "div"] {
            assert!(
                !names.contains(&op),
                "{op:?} is spelled with an operator, not a method call"
            );
        }
    }

    #[test]
    fn excludes_lowering_only_bit_manipulation_primitives() {
        let names: Vec<&str> = known_method_names().collect();
        for op in [
            "trunc_to_int",
            "int_to_float",
            "iadd",
            "shl",
            "shr",
            "bitand",
            "bitor",
        ] {
            assert!(
                !names.contains(&op),
                "{op:?} only arises from lowering passes, never surface syntax"
            );
        }
    }
}

#[cfg(test)]
mod from_method_call {
    use super::OpKind;

    #[test]
    fn resolves_a_unary_method_at_zero_args() {
        assert_eq!(OpKind::from_method_call("sqrt", 0), Some(OpKind::Sqrt));
    }

    #[test]
    fn resolves_a_binary_method_at_one_arg() {
        assert_eq!(OpKind::from_method_call("min", 1), Some(OpKind::Min));
    }

    #[test]
    fn resolves_a_ternary_method_at_two_args() {
        assert_eq!(OpKind::from_method_call("mul_add", 2), Some(OpKind::MulAdd));
        assert_eq!(OpKind::from_method_call("select", 2), Some(OpKind::Select));
    }

    #[test]
    fn resolves_neg_as_a_method_despite_its_operator_emit_style() {
        assert_eq!(OpKind::from_method_call("neg", 0), Some(OpKind::Neg));
    }

    #[test]
    fn accepts_the_powf_alias_for_pow() {
        assert_eq!(OpKind::from_method_call("powf", 1), Some(OpKind::Pow));
    }

    #[test]
    fn rejects_a_wrong_arg_count() {
        assert_eq!(OpKind::from_method_call("sqrt", 1), None);
        assert_eq!(OpKind::from_method_call("min", 0), None);
        assert_eq!(OpKind::from_method_call("min", 2), None);
    }

    #[test]
    fn rejects_infix_arithmetic() {
        assert_eq!(OpKind::from_method_call("add", 1), None);
    }

    #[test]
    fn rejects_lowering_only_primitives() {
        assert_eq!(OpKind::from_method_call("trunc_to_int", 0), None);
        assert_eq!(OpKind::from_method_call("shl", 1), None);
    }

    #[test]
    fn rejects_an_unrecognized_name() {
        assert_eq!(OpKind::from_method_call("not_a_real_op", 0), None);
    }
}

#[cfg(test)]
mod eval_unary_libm_arms {
    use super::OpKind;

    #[test]
    fn ceil_rounds_toward_positive_infinity() {
        assert_eq!(OpKind::Ceil.eval_unary(1.2), Some(2.0));
        assert_eq!(OpKind::Ceil.eval_unary(-1.2), Some(-1.0));
    }

    #[test]
    fn round_ties_to_even_matching_x86s_vroundps() {
        assert_eq!(OpKind::Round.eval_unary(2.5), Some(2.0));
        assert_eq!(OpKind::Round.eval_unary(3.5), Some(4.0));
        assert_eq!(OpKind::Round.eval_unary(1.2), Some(1.0));
    }

    #[test]
    fn int_to_float_reinterprets_bits_rather_than_converting_the_value() {
        // 1.0f32's bit pattern read as i32 is nowhere near 1.0.
        let one_bits = 1.0f32.to_bits() as i32;
        assert_eq!(
            OpKind::IntToFloat.eval_unary(1.0),
            Some(one_bits as f32),
            "IntToFloat must reinterpret x's bits as i32, not convert x's value"
        );
    }

    #[test]
    fn recip_is_exact_reciprocal_not_an_estimate() {
        assert_eq!(OpKind::Recip.eval_unary(4.0), Some(0.25));
    }
}

/// Independently-authored oracles for the algebraic-property predicates,
/// exhaustive over `OpKind::all()`. Each predicate is a total function over a
/// closed enum, so — unlike a spot check — a single exhaustive sweep is
/// enough to catch both "flipped the wrong op" and "always returns the same
/// answer" mistakes.
#[cfg(test)]
mod algebraic_properties {
    use super::OpKind;

    const COMMUTATIVE: &[OpKind] = &[
        OpKind::Add,
        OpKind::Mul,
        OpKind::Min,
        OpKind::Max,
        OpKind::Eq,
        OpKind::Ne,
    ];

    #[test]
    fn flag_add_mul_min_max_eq_ne_as_commutative_and_no_other_op() {
        for op in OpKind::all() {
            assert_eq!(
                op.is_commutative(),
                COMMUTATIVE.contains(&op),
                "{op:?} commutativity mismatch"
            );
        }
    }

    const ASSOCIATIVE: &[OpKind] = &[OpKind::Add, OpKind::Mul, OpKind::Min, OpKind::Max];

    #[test]
    fn flag_add_mul_min_max_as_associative_and_no_other_op() {
        for op in OpKind::all() {
            assert_eq!(
                op.is_associative(),
                ASSOCIATIVE.contains(&op),
                "{op:?} associativity mismatch"
            );
        }
    }

    #[test]
    fn return_zero_for_add_sub_one_for_mul_div_and_none_otherwise() {
        for op in OpKind::all() {
            let want = match op {
                OpKind::Add | OpKind::Sub => Some(0.0),
                OpKind::Mul | OpKind::Div => Some(1.0),
                _ => None,
            };
            assert_eq!(op.identity(), want, "{op:?} identity mismatch");
        }
    }

    #[test]
    fn return_zero_annihilator_only_for_mul() {
        for op in OpKind::all() {
            let want = if op == OpKind::Mul { Some(0.0) } else { None };
            assert_eq!(op.annihilator(), want, "{op:?} annihilator mismatch");
        }
    }

    const IDEMPOTENT: &[OpKind] = &[OpKind::Min, OpKind::Max, OpKind::Abs];

    #[test]
    fn flag_min_max_abs_as_idempotent_and_no_other_op() {
        for op in OpKind::all() {
            assert_eq!(
                op.is_idempotent(),
                IDEMPOTENT.contains(&op),
                "{op:?} idempotence mismatch"
            );
        }
    }

    const NOT_SEED_OPS: &[OpKind] = &[
        OpKind::Var,
        OpKind::Const,
        OpKind::Tuple,
        OpKind::MulAdd,
        OpKind::Lt,
        OpKind::Le,
        OpKind::Gt,
        OpKind::Ge,
        OpKind::Eq,
        OpKind::Ne,
        OpKind::Select,
        OpKind::Buffer,
        OpKind::Gather,
        OpKind::RawGather,
        OpKind::Reduce,
        OpKind::Uniform,
        OpKind::Param,
    ];

    #[test]
    fn exclude_leaves_masks_and_memory_ops_from_seed_ops() {
        for op in OpKind::all() {
            assert_eq!(
                op.is_seed_op(),
                !NOT_SEED_OPS.contains(&op),
                "{op:?} seed-op eligibility mismatch"
            );
        }
    }

    const BITWISE_DOMAIN: &[OpKind] = &[
        OpKind::Lt,
        OpKind::Le,
        OpKind::Gt,
        OpKind::Ge,
        OpKind::Eq,
        OpKind::Ne,
        OpKind::Select,
        OpKind::BitAnd,
        OpKind::BitOr,
        OpKind::TruncToInt,
        OpKind::IntToFloat,
        OpKind::IAdd,
        OpKind::Shl,
        OpKind::Shr,
    ];

    #[test]
    fn flag_masks_bitops_and_integer_primitives_as_bitwise_domain() {
        for op in OpKind::all() {
            assert_eq!(
                op.is_bitwise_domain(),
                BITWISE_DOMAIN.contains(&op),
                "{op:?} bitwise-domain mismatch"
            );
        }
    }

    #[test]
    fn match_each_ops_actual_operand_count() {
        for op in OpKind::all() {
            let want = match op {
                OpKind::Var
                | OpKind::Const
                | OpKind::Tuple
                | OpKind::Buffer
                | OpKind::Uniform
                | OpKind::Param => 0,

                OpKind::Neg
                | OpKind::Sqrt
                | OpKind::Rsqrt
                | OpKind::Abs
                | OpKind::Recip
                | OpKind::Floor
                | OpKind::Ceil
                | OpKind::Round
                | OpKind::Sin
                | OpKind::Cos
                | OpKind::Tan
                | OpKind::Asin
                | OpKind::Acos
                | OpKind::Atan
                | OpKind::Exp
                | OpKind::Exp2
                | OpKind::Ln
                | OpKind::Log2
                | OpKind::Log10
                | OpKind::TruncToInt
                | OpKind::IntToFloat => 1,

                OpKind::Add
                | OpKind::Sub
                | OpKind::Mul
                | OpKind::Div
                | OpKind::Min
                | OpKind::Max
                | OpKind::Atan2
                | OpKind::Pow
                | OpKind::Lt
                | OpKind::Le
                | OpKind::Gt
                | OpKind::Ge
                | OpKind::Eq
                | OpKind::Ne
                | OpKind::IAdd
                | OpKind::Shl
                | OpKind::Shr
                | OpKind::BitAnd
                | OpKind::BitOr
                | OpKind::Dwrt
                | OpKind::RawGather => 2,

                OpKind::MulAdd | OpKind::Select | OpKind::Gather => 3,

                OpKind::Reduce => 1,
            };
            assert_eq!(op.arity(), want, "{op:?} arity mismatch");
        }
    }

    #[test]
    fn distinguish_leaf_arithmetic_memory_and_transcendental_costs() {
        // Not exhaustive (unlike the predicates above): default_cost's price
        // table is a design choice, not a closed algebraic property, so a few
        // ops from each priced tier is enough to catch "the whole function
        // got replaced by a constant" without freezing every literal in the
        // table against future re-tuning.
        assert_eq!(OpKind::Var.default_cost(), 0);
        assert_eq!(OpKind::Neg.default_cost(), 1);
        assert_eq!(OpKind::Add.default_cost(), 4);
        assert_eq!(OpKind::Mul.default_cost(), 5);
        assert_eq!(OpKind::Gather.default_cost(), 10);
        assert_eq!(OpKind::Sin.default_cost(), 15);
        assert_eq!(
            OpKind::Dwrt.default_cost(),
            1_000_000,
            "Dwrt must outprice every real op so a surviving one is never preferred"
        );
    }
}

#[cfg(test)]
mod from_name {
    use super::OpKind;

    #[test]
    fn round_trip_every_ops_own_name_including_special_emit_style_ops() {
        // `method_names::every_returned_name_round_trips_through_from_name`
        // only walks `known_method_names()`, which deliberately excludes
        // every `EmitStyle::Special` op (Var/Const/Tuple/Dwrt/Buffer/Gather/
        // RawGather/Reduce) — so those ops' `from_name` arms need their own
        // coverage here.
        for op in OpKind::all() {
            assert_eq!(
                OpKind::from_name(op.name()),
                Some(op),
                "{op:?}'s own name does not parse back to itself"
            );
        }
    }

    #[test]
    fn accept_the_powf_alias_for_pow() {
        assert_eq!(OpKind::from_name("powf"), Some(OpKind::Pow));
    }

    #[test]
    fn reject_an_unrecognized_name() {
        assert_eq!(OpKind::from_name("not_a_real_op"), None);
    }
}

#[cfg(test)]
mod eval_binary_arms {
    use super::OpKind;

    /// Compares by bit pattern, not `==` — a true mask is the all-ones NaN
    /// pattern, and NaN never equals itself under `f32::eq`.
    fn assert_mask(got: Option<f32>, want: bool) {
        assert_eq!(
            got.expect("op is binary").to_bits(),
            OpKind::mask(want).to_bits()
        );
    }

    #[test]
    fn compute_ordinary_arithmetic_for_add_sub_mul_div() {
        assert_eq!(OpKind::Add.eval_binary(2.0, 3.0), Some(5.0));
        assert_eq!(OpKind::Sub.eval_binary(5.0, 3.0), Some(2.0));
        assert_eq!(OpKind::Mul.eval_binary(2.0, 3.0), Some(6.0));
        assert_eq!(OpKind::Div.eval_binary(6.0, 3.0), Some(2.0));
    }

    #[test]
    fn pick_the_second_operand_for_min_and_max_on_a_tie() {
        // x86 minps/maxps: `(a OP b) ? a : b`, so equal operands return the
        // SECOND one — see `fold_is_platform_specific`'s doc. `0.0` and
        // `-0.0` are the tie that can actually prove this: they compare
        // equal under `<`/`>` but carry different bit patterns, so unlike a
        // same-value tie (e.g. `1.0, 1.0`) the result reveals which operand
        // was returned instead of leaving it ambiguous.
        assert_eq!(
            OpKind::Min.eval_binary(0.0, -0.0).unwrap().to_bits(),
            (-0.0f32).to_bits(),
        );
        assert_eq!(
            OpKind::Min.eval_binary(-0.0, 0.0).unwrap().to_bits(),
            0.0f32.to_bits(),
        );
        assert_eq!(
            OpKind::Max.eval_binary(0.0, -0.0).unwrap().to_bits(),
            (-0.0f32).to_bits(),
        );
        assert_eq!(
            OpKind::Max.eval_binary(-0.0, 0.0).unwrap().to_bits(),
            0.0f32.to_bits(),
        );
        assert_eq!(OpKind::Min.eval_binary(1.0, 2.0), Some(1.0));
        assert_eq!(OpKind::Min.eval_binary(2.0, 1.0), Some(1.0));
        assert_eq!(OpKind::Max.eval_binary(1.0, 2.0), Some(2.0));
        assert_eq!(OpKind::Max.eval_binary(2.0, 1.0), Some(2.0));
    }

    #[test]
    fn agree_on_strict_and_equal_operands_for_lt_and_le() {
        assert_mask(OpKind::Lt.eval_binary(1.0, 2.0), true);
        assert_mask(OpKind::Lt.eval_binary(2.0, 2.0), false);
        assert_mask(OpKind::Le.eval_binary(2.0, 2.0), true);
        assert_mask(OpKind::Le.eval_binary(3.0, 2.0), false);
    }

    #[test]
    fn treat_gt_and_ge_as_unordered_true_for_a_nan_operand() {
        // x86's imm8 6/5 (NLE_US/NLT_US): NaN in EITHER operand is TRUE.
        assert_mask(OpKind::Gt.eval_binary(f32::NAN, 1.0), true);
        assert_mask(OpKind::Gt.eval_binary(1.0, f32::NAN), true);
        assert_mask(OpKind::Gt.eval_binary(2.0, 1.0), true);
        assert_mask(OpKind::Gt.eval_binary(1.0, 2.0), false);
        assert_mask(OpKind::Gt.eval_binary(1.0, 1.0), false);
        assert_mask(OpKind::Ge.eval_binary(f32::NAN, 1.0), true);
        assert_mask(OpKind::Ge.eval_binary(1.0, f32::NAN), true);
        assert_mask(OpKind::Ge.eval_binary(1.0, 1.0), true);
        assert_mask(OpKind::Ge.eval_binary(1.0, 2.0), false);
    }

    #[test]
    fn treat_eq_and_ne_as_exact_and_disagree_only_on_nan() {
        assert_mask(OpKind::Eq.eval_binary(1.0, 1.0), true);
        assert_mask(OpKind::Eq.eval_binary(1.0, 2.0), false);
        assert_mask(OpKind::Eq.eval_binary(f32::NAN, f32::NAN), false);
        assert_mask(OpKind::Ne.eval_binary(1.0, 2.0), true);
        assert_mask(OpKind::Ne.eval_binary(1.0, 1.0), false);
        assert_mask(OpKind::Ne.eval_binary(f32::NAN, f32::NAN), true);
    }

    #[test]
    fn wrap_the_bit_pattern_as_a_two_s_complement_integer_for_iadd() {
        let got = OpKind::IAdd
            .eval_binary(f32::from_bits(i32::MAX as u32), f32::from_bits(1))
            .unwrap();
        assert_eq!(got.to_bits(), i32::MIN as u32);
    }

    #[test]
    fn mask_the_count_to_five_bits_before_shifting() {
        // count=33 masks to 1 (33 & 31 == 1) — same as an explicit shift of 1.
        let one_bit = f32::from_bits(1);
        assert_eq!(OpKind::Shl.eval_binary(one_bit, 33.0).unwrap().to_bits(), 2);

        let hi_bit = f32::from_bits(1u32 << 31);
        assert_eq!(
            OpKind::Shr.eval_binary(hi_bit, 33.0).unwrap().to_bits(),
            1u32 << 30
        );
    }

    #[test]
    fn operate_on_raw_bit_patterns_for_bitand_and_bitor() {
        let a = f32::from_bits(0b1010);
        let b = f32::from_bits(0b0110);
        assert_eq!(OpKind::BitAnd.eval_binary(a, b).unwrap().to_bits(), 0b0010);
        assert_eq!(OpKind::BitOr.eval_binary(a, b).unwrap().to_bits(), 0b1110);
    }

    #[test]
    fn return_none_from_eval_binary_for_a_unary_op() {
        assert_eq!(OpKind::Neg.eval_binary(1.0, 2.0), None);
    }
}

#[cfg(test)]
mod eval_ternary_arms {
    use super::OpKind;

    #[test]
    fn compute_x_times_y_plus_z_for_mul_add() {
        assert_eq!(OpKind::MulAdd.eval_ternary(2.0, 3.0, 4.0), Some(10.0));
    }

    #[test]
    fn blend_y_and_z_bitwise_by_the_x_mask_for_select() {
        let true_mask = OpKind::mask(true);
        let false_mask = OpKind::mask(false);
        let y = 7.0f32;
        let z = 9.0f32;
        assert_eq!(
            OpKind::Select
                .eval_ternary(true_mask, y, z)
                .unwrap()
                .to_bits(),
            y.to_bits()
        );
        assert_eq!(
            OpKind::Select
                .eval_ternary(false_mask, y, z)
                .unwrap()
                .to_bits(),
            z.to_bits()
        );
    }

    #[test]
    fn return_none_from_eval_ternary_for_a_binary_op() {
        assert_eq!(OpKind::Add.eval_ternary(1.0, 2.0, 3.0), None);
    }
}

#[cfg(test)]
mod fold_is_platform_specific {
    use super::OpKind;

    #[test]
    fn decline_to_fold_min_and_max_on_nan_or_a_signed_zero_mismatch() {
        assert!(OpKind::Min.fold_is_platform_specific(&[f32::NAN, 1.0]));
        assert!(OpKind::Min.fold_is_platform_specific(&[1.0, f32::NAN]));
        assert!(OpKind::Max.fold_is_platform_specific(&[f32::NAN, 1.0]));
        assert!(OpKind::Min.fold_is_platform_specific(&[0.0, -0.0]));
        assert!(OpKind::Max.fold_is_platform_specific(&[-0.0, 0.0]));
        assert!(!OpKind::Min.fold_is_platform_specific(&[1.0, 2.0]));
        assert!(!OpKind::Max.fold_is_platform_specific(&[0.0, 0.0]));
        // No [a, b] pattern applies to the wrong arity, so it falls to `false`.
        assert!(!OpKind::Min.fold_is_platform_specific(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn decline_to_fold_gt_and_ge_only_when_an_operand_is_nan() {
        assert!(OpKind::Gt.fold_is_platform_specific(&[f32::NAN, 1.0]));
        assert!(OpKind::Ge.fold_is_platform_specific(&[1.0, f32::NAN]));
        assert!(!OpKind::Gt.fold_is_platform_specific(&[1.0, 2.0]));
        assert!(!OpKind::Ge.fold_is_platform_specific(&[1.0, 2.0]));
    }

    #[test]
    fn decline_to_fold_round_at_any_exact_tie_and_in_the_negative_zero_interval() {
        assert!(OpKind::Round.fold_is_platform_specific(&[2.5])); // positive tie
        assert!(OpKind::Round.fold_is_platform_specific(&[-0.5])); // negative tie
        assert!(OpKind::Round.fold_is_platform_specific(&[-0.25])); // in (-0.5, -0.0]
        assert!(!OpKind::Round.fold_is_platform_specific(&[2.3])); // ordinary, no tie
        assert!(!OpKind::Round.fold_is_platform_specific(&[-0.6])); // outside the interval
    }

    #[test]
    fn decline_to_fold_trunc_to_int_on_nan_and_values_at_or_above_two_pow_31() {
        assert!(OpKind::TruncToInt.fold_is_platform_specific(&[f32::NAN]));
        assert!(OpKind::TruncToInt.fold_is_platform_specific(&[2_147_483_648.0]));
        // 2^31 - 128: the nearest f32 below the limit that is exactly
        // representable (2^31 - 1 itself rounds UP to 2^31 as an f32 literal).
        assert!(!OpKind::TruncToInt.fold_is_platform_specific(&[2_147_483_520.0]));
        // Both tiers agree in the negative-overflow direction — see the
        // function's doc for why.
        assert!(!OpKind::TruncToInt.fold_is_platform_specific(&[-2_147_483_648.0]));
        assert!(!OpKind::TruncToInt.fold_is_platform_specific(&[f32::NEG_INFINITY]));
    }

    #[test]
    fn decline_to_fold_shl_and_shr_on_a_count_outside_0_to_32_or_non_integral() {
        assert!(OpKind::Shl.fold_is_platform_specific(&[1.0, 32.0]));
        assert!(OpKind::Shr.fold_is_platform_specific(&[1.0, -1.0]));
        assert!(OpKind::Shl.fold_is_platform_specific(&[1.0, 1.5]));
        assert!(!OpKind::Shl.fold_is_platform_specific(&[1.0, 5.0]));
        assert!(!OpKind::Shr.fold_is_platform_specific(&[1.0, 0.0]));
    }

    #[test]
    fn always_fold_mul_add_and_round_it_once() {
        // The inputs where one rounding and two disagree are the whole point:
        // that gap is precision, inside the contract, and never blocks a fold.
        let (a, b, c) = (1.0000001f32, 4097.0, 4097.0);
        assert_ne!(
            libm::fmaf(a, b, c).to_bits(),
            (core::hint::black_box(a * b) + c).to_bits()
        );
        assert!(!OpKind::MulAdd.fold_is_platform_specific(&[a, b, c]));
        assert_eq!(
            OpKind::MulAdd.eval_ternary(a, b, c).map(f32::to_bits),
            Some(libm::fmaf(a, b, c).to_bits())
        );
    }

    #[test]
    fn always_decline_to_fold_recip_and_rsqrt() {
        assert!(OpKind::Recip.fold_is_platform_specific(&[2.0]));
        assert!(OpKind::Rsqrt.fold_is_platform_specific(&[4.0]));
    }

    #[test]
    fn never_decline_to_fold_an_ordinary_op_like_add_or_lt() {
        assert!(!OpKind::Add.fold_is_platform_specific(&[1.0, 2.0]));
        assert!(!OpKind::Lt.fold_is_platform_specific(&[1.0, 2.0]));
    }
}
