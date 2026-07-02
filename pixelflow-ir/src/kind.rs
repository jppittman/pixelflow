//! Operation Kind Enumeration.
//!
//! This enum provides a uniform representation of all operations.
//! It is used for storage in the e-graph and as the base for feature indices.

use crate::traits::EmitStyle;

/// Unified enumeration of all IR operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum OpKind {
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
    Fract = 17,

    // --- Trigonometry ---
    Sin = 18,
    Cos = 19,
    Tan = 20,
    Asin = 21,
    Acos = 22,
    Atan = 23,
    Atan2 = 24,

    // --- Exponentials ---
    Exp = 25,
    Exp2 = 26,
    Ln = 27,
    Log2 = 28,
    Log10 = 29,
    Pow = 30,
    Hypot = 31,

    // --- Comparison ---
    Lt = 32,
    Le = 33,
    Gt = 34,
    Ge = 35,
    Eq = 36,
    Ne = 37,

    // --- Control Flow ---
    Select = 38,
    Clamp = 39,

    // --- Structure ---
    Tuple = 40,

    // --- Bit manipulation (integer-domain primitives) ---
    // These let exp/ln/log lower to arithmetic: each maps 1:1 to a single
    // hardware instruction. Lanes are reinterpreted as i32 where noted;
    // "reinterpret" itself is free (same register) and is not an op.
    /// Truncate f32 lanes to i32 (`cvttps2dq` / `fcvtzs`).
    TruncToInt = 41,
    /// Convert i32 lanes to f32 (`cvtdq2ps` / `scvtf`).
    IntToFloat = 42,
    /// Integer (i32) lane-wise add (`paddd` / `add .4s`).
    IAdd = 43,
    /// Logical shift-left of i32 lanes; RHS is a `Const` shift amount.
    Shl = 44,
    /// Logical shift-right of i32 lanes; RHS is a `Const` shift amount.
    Shr = 45,
    /// Bitwise AND of lane bit patterns (`andps` / `and`).
    BitAnd = 46,
    /// Bitwise OR of lane bit patterns (`orps` / `orr`).
    BitOr = 47,

    // --- Differentiation ---
    /// Symbolic derivative `∂(child0)/∂(var)` where `child1` is a `Const` whose
    /// value is the variable index (0=X, 1=Y, 2=Z, 3=W). The e-graph pushes
    /// `Dwrt` toward the leaves via the chain rule, computing the derivative
    /// analytically; whatever cannot be decomposed survives as a residual `Dwrt`
    /// (the jet fallback — not yet wired). It must never reach a backend.
    Dwrt = 48,

    // --- Bound memory (lattices) ---
    /// Buffer leaf: a slot referencing a `BufferDecl` in the arena's buffer
    /// table. The declared extents are static IR; the contents are bound at
    /// JIT-compile time. See `docs/designs/KERNELS_AND_LATTICES.md`.
    Buffer = 49,
    /// Read a bound buffer: `Gather(buffer, x, y)` where `buffer` is a
    /// `Buffer` leaf. Semantics match `DiscreteManifold::eval`: floor the
    /// indices, clamp to the declared extents, gather row-major.
    Gather = 50,
    /// Primitive gather: `RawGather(buffer, index)` reads `buffer`'s contents
    /// at the already-computed linear lane `index` (truncated to int), with no
    /// floor/clamp/row-major math. `Gather` lowers to index arithmetic (built
    /// from existing ops) plus this primitive — the analogue of `raw_mul` under
    /// `mul`. The index is trusted to be in bounds (the lowering clamps it).
    RawGather = 51,

    // --- Reduction (lattice fold) ---
    /// Fold a body over a bounded domain. Encoded
    /// `Nary(Reduce, [Const(combiner), Const(reduce_var), Const(extent), body])`:
    /// `combiner` is the monoid op index (`Add`/`Mul`/`Min`/`Max`), `body`
    /// references `Var(reduce_var)` (indices 4..8), and the fold runs over
    /// `0..extent`. The combiner is a *child* (a parameter), not baked into the
    /// opcode, so one `Reduce` covers every monoid and can later take an
    /// arbitrary combiner function. Lowered to an unrolled accumulation by
    /// `expand_reduce` before codegen — the analogue of `Gather -> RawGather`.
    Reduce = 52,
}

impl OpKind {
    /// Total number of operations.
    pub const COUNT: usize = 53;

    /// Monoid identity for an op usable as a reduction combiner
    /// (`Add`→0, `Mul`→1, `Min`→+∞, `Max`→−∞). `None` if `self` is not a valid
    /// combiner.
    #[must_use]
    pub const fn monoid_identity(self) -> Option<f32> {
        match self {
            Self::Add => Some(0.0),
            Self::Mul => Some(1.0),
            Self::Min => Some(f32::INFINITY),
            Self::Max => Some(f32::NEG_INFINITY),
            _ => None,
        }
    }

    /// Whether `self` is a valid reduction combiner (an associative monoid op
    /// with an identity).
    #[must_use]
    pub const fn is_monoid(self) -> bool {
        self.monoid_identity().is_some()
    }

    /// Convert to array index.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Convert index to OpKind.
    #[must_use]
    pub fn from_index(idx: usize) -> Option<Self> {
        if idx >= Self::COUNT {
            return None;
        }
        // SAFETY: repr(u8) and contiguous 0..=40
        unsafe { core::mem::transmute(idx as u8) }
    }

    /// Get the arity of the operation.
    #[must_use]
    pub const fn arity(self) -> usize {
        match self {
            Self::Var | Self::Const | Self::Tuple | Self::Buffer => 0,

            Self::Neg
            | Self::Sqrt
            | Self::Rsqrt
            | Self::Abs
            | Self::Recip
            | Self::Floor
            | Self::Ceil
            | Self::Round
            | Self::Fract
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
            | Self::Hypot
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

            Self::MulAdd | Self::Select | Self::Clamp | Self::Gather => 3,

            // N-ary: [combiner, reduce_var, extent, body].
            Self::Reduce => 4,
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
            Self::Fract => "fract",
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
            Self::Hypot => "hypot",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Select => "select",
            Self::Clamp => "clamp",
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
            "fract" => Some(Self::Fract),
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
            "hypot" => Some(Self::Hypot),
            "lt" => Some(Self::Lt),
            "le" => Some(Self::Le),
            "gt" => Some(Self::Gt),
            "ge" => Some(Self::Ge),
            "eq" => Some(Self::Eq),
            "ne" => Some(Self::Ne),
            "select" => Some(Self::Select),
            "clamp" => Some(Self::Clamp),
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
            _ => None,
        }
    }

    /// Get the default cost estimate for this operation (in cycles).
    #[must_use]
    pub const fn default_cost(self) -> usize {
        match self {
            Self::Var | Self::Const | Self::Tuple | Self::Buffer => 0,
            // Memory read: native gather on AVX2/AVX-512, scalar loads on
            // NEON/SSE2. Priced between an arithmetic op and a transcendental.
            Self::Gather | Self::RawGather => 10,
            // Reduction is lowered (unrolled) away before costing; price the
            // node itself at zero so a stray one never dominates extraction.
            Self::Reduce => 0,
            Self::Neg | Self::Abs | Self::Floor | Self::Ceil | Self::Round | Self::Fract => 1,
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
            | Self::Select
            | Self::Clamp => 4,
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
            | Self::Pow
            | Self::Hypot => 15,
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
        )
    }

    /// Get the emit style for code generation.
    #[must_use]
    pub const fn emit_style(self) -> EmitStyle {
        match self {
            // Special cases handled separately
            Self::Var | Self::Const | Self::Tuple => EmitStyle::Special,

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
            | Self::Fract
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
            | Self::Hypot
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

            // Memory ops: emitted by the JIT binding path, not as method calls.
            Self::Buffer | Self::Gather | Self::RawGather => EmitStyle::Special,

            // Reduction: lowered to unrolled arithmetic before codegen.
            Self::Reduce => EmitStyle::Special,

            // Ternary method: (a).mul_add(b, c)
            Self::MulAdd | Self::Select | Self::Clamp => EmitStyle::TernaryMethod,
        }
    }

    // NOTE: KNOWN_METHODS was removed - now derived from ALL_OPS via known_method_names()

    /// Evaluate a unary operation on a constant argument.
    ///
    /// Returns `None` for non-unary operations or operations that can't be
    /// evaluated at compile time.
    #[must_use]
    pub fn eval_unary(self, x: f32) -> Option<f32> {
        match self {
            Self::Neg => Some(-x),
            Self::Sqrt => Some(x.sqrt()),
            Self::Rsqrt => Some(1.0 / x.sqrt()),
            Self::Abs => Some(x.abs()),
            Self::Recip => Some(1.0 / x),
            Self::Floor => Some(x.floor()),
            Self::Ceil => Some(x.ceil()),
            Self::Round => Some(x.round()),
            Self::Fract => Some(x.fract()),
            Self::Sin => Some(x.sin()),
            Self::Cos => Some(x.cos()),
            Self::Tan => Some(x.tan()),
            Self::Asin => Some(x.asin()),
            Self::Acos => Some(x.acos()),
            Self::Atan => Some(x.atan()),
            Self::Exp => Some(x.exp()),
            Self::Exp2 => Some(x.exp2()),
            Self::Ln => Some(x.ln()),
            Self::Log2 => Some(x.log2()),
            Self::Log10 => Some(x.log10()),
            _ => None,
        }
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
            Self::Min => Some(x.min(y)),
            Self::Max => Some(x.max(y)),
            Self::Atan2 => Some(x.atan2(y)),
            Self::Pow => Some(x.powf(y)),
            Self::Hypot => Some(x.hypot(y)),
            Self::Lt => Some(if x < y { 1.0 } else { 0.0 }),
            Self::Le => Some(if x <= y { 1.0 } else { 0.0 }),
            Self::Gt => Some(if x > y { 1.0 } else { 0.0 }),
            Self::Ge => Some(if x >= y { 1.0 } else { 0.0 }),
            Self::Eq => Some(if (x - y).abs() < f32::EPSILON {
                1.0
            } else {
                0.0
            }),
            Self::Ne => Some(if (x - y).abs() >= f32::EPSILON {
                1.0
            } else {
                0.0
            }),
            _ => None,
        }
    }

    /// Evaluate a ternary operation on constant arguments.
    #[must_use]
    pub fn eval_ternary(self, x: f32, y: f32, z: f32) -> Option<f32> {
        match self {
            Self::MulAdd => Some(x * y + z),
            Self::Select => Some(if x != 0.0 { y } else { z }),
            Self::Clamp => {
                // Guard against degenerate clamp where min > max
                // (can arise from e-graph extraction with swapped bounds).
                let lo = y.min(z);
                let hi = y.max(z);
                Some(x.clamp(lo, hi))
            }
            _ => None,
        }
    }
}

// EmitStyle is imported from crate::traits - single source of truth
