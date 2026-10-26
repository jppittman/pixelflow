//! Which operations an e-graph may hold, and which a rewrite template may name.
//!
//! An operation is an [`OpKind`]; its algebraic properties (`identity`,
//! `annihilator`, `is_commutative`, ...) are `OpKind`'s own, the single source
//! of truth. This module holds only the two admission policies, because they
//! are the part that is the e-graph's rather than the instruction set's.

use pixelflow_ir::OpKind;

/// Whether a rewrite template may *name* `kind`.
///
/// Always the smaller set of the two admission policies: [`Vocabulary::admits`]
/// adds to it. The match is exhaustive on purpose, so a new `OpKind` fails the
/// build here until someone decides which side of the line it is on.
///
/// - Leaves (`Var`, `Const`, `Buffer`, `Uniform`, `Param`) are not operations.
/// - `Dwrt(expr, var)` is the single autodiff operator. It exists only inside
///   the e-graph: chain-rule rewrites push it toward the leaves until it
///   dissolves into ordinary arithmetic. A surviving `Dwrt` after saturation is
///   the jet fallback (not yet wired) and carries a prohibitive cost so the
///   extractor never prefers it.
/// - `Gather` is representable (`ENode::Buffer` carries the full `BufferDecl`)
///   but stays unnameable: its semantics depend on bound memory, so no
///   algebraic rule, and no constant fold, may look through it. Denying every
///   template the ability to name it leaves its participation to hash-consing
///   CSE.
/// - The mask and integer-domain ops are the runtime tier's alone (see
///   [`Vocabulary::admits`]).
/// - `RawGather` and `Seq` are built after the e-graph runs; sequencing is an
///   effect, so no rule may name it and no e-graph holds it.
/// - `Reduce` is held as its own node, `ENode::Reduce`, which carries its fold
///   header as a parameter; it is not an op over operands that a rule could name.
/// - `Tuple` has no fixed arity, which an inline-operand node cannot hold.
/// - `Shl` and `Shr` are held as `ENode::Shift`, which carries the count as a
///   parameter; they are not operations over operand classes, so no rule
///   could name one.
/// - `Bits` and `Float`, the reinterpretations, are integer-domain ops like
///   the rest and stay with them: runtime tier only, and nameable by no
///   template. Their two identities (`Float(Bits(x)) = x` and the converse)
///   are not rules, since naming them here would admit the ops to the macro
///   tier as well; they are held opaque, and the pair compiles to nothing
///   either way (`program/lower.rs`).
#[must_use]
pub fn template_may_name(kind: OpKind) -> bool {
    match kind {
        OpKind::Add
        | OpKind::Sub
        | OpKind::Mul
        | OpKind::Div
        | OpKind::Neg
        | OpKind::Recip
        | OpKind::Sqrt
        | OpKind::Rsqrt
        | OpKind::Abs
        | OpKind::Min
        | OpKind::Max
        | OpKind::MulAdd
        | OpKind::Floor
        | OpKind::Ceil
        | OpKind::Round
        | OpKind::Sin
        | OpKind::Cos
        | OpKind::Tan
        | OpKind::Asin
        | OpKind::Acos
        | OpKind::Atan
        | OpKind::Atan2
        | OpKind::Exp
        | OpKind::Exp2
        | OpKind::Ln
        | OpKind::Log2
        | OpKind::Log10
        | OpKind::Pow
        | OpKind::Lt
        | OpKind::Le
        | OpKind::Gt
        | OpKind::Ge
        | OpKind::Eq
        | OpKind::Ne
        | OpKind::If
        | OpKind::Dwrt => true,
        OpKind::Var
        | OpKind::Const
        | OpKind::Buffer
        | OpKind::Uniform
        | OpKind::Param
        | OpKind::Gather
        | OpKind::RawGather
        | OpKind::Reduce
        | OpKind::Seq
        | OpKind::Tuple
        | OpKind::TruncToInt
        | OpKind::IntToFloat
        | OpKind::IAdd
        | OpKind::Shl
        | OpKind::Shr
        | OpKind::BitAnd
        | OpKind::BitOr
        | OpKind::Bits
        | OpKind::Float => false,
    }
}

/// Which ops an e-graph is allowed to *hold*, as opposed to which ops a
/// rewrite template may *name* ([`template_may_name`], always the smaller set).
///
/// The two differ, and the difference is per-tier rather than global — which
/// is the whole reason this is an argument. Mask and integer-domain ops are
/// sound in a runtime-tier graph (built at bake time, after composition, where
/// the derivative calculus has its full context) and unsound in a macro-tier
/// one (expanded before composition, where a leaf's `DX` is not yet scaled by
/// an enclosing warp; the fonts' density-dependent AA ramp broke exactly this
/// way when these ops were briefly global). That was previously encoded in
/// *which private helper a call site happened to call*, so nothing checked it.
/// Naming the vocabulary makes the choice visible at each insertion, and wrong
/// only by writing the wrong variant.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Vocabulary {
    /// What a rewrite template can name, plus the opaque `Gather`. The
    /// macro (`kernel!`) and `Dwrt` expansion tiers.
    Templates,
    /// [`Self::Templates`] plus the mask and integer-domain ops. The runtime
    /// tier, which optimizes composed arenas at bake time.
    Runtime,
}

impl Vocabulary {
    /// Whether a graph under this vocabulary may hold `kind`.
    ///
    /// The runtime tier's extras:
    ///
    /// - `&`/`|` on comparison masks are surface-language ops (every glyph
    ///   winding kernel's Y-range gate is `(Y >= lo) & (Y < hi)`), so
    ///   runtime-built arenas must be representable with them present.
    /// - The packed cell-grid kernel's spine, clamp → `TruncToInt` → `Shl` →
    ///   or-fold, builds a `u32` pixel per lane; the production frame kernel
    ///   compiles with no CSE across its four channels unless these enter the
    ///   e-graph.
    ///
    /// No rewrite rule targets any of them, but admission is not fold-opacity:
    /// `ConstantFold` reads any `ENode::Op` or `ENode::Shift`, so every admitted
    /// op folds and each needs its own answer to "does this fold agree with
    /// what the backends emit?" `OpKind::fold_is_platform_specific` is where
    /// that answer lives; being unnameable by a template guards nothing.
    ///
    /// `Shl`/`Shr` are held as `ENode::Shift`, whose count is a `ShiftCount`:
    /// [`insert`](fn@super::insert) declines a shift whose count is not a
    /// literal [`ShiftCount`](pixelflow_ir::ShiftCount), and extraction
    /// writes the count back as the `Const` the emitter's immediate-only
    /// contract needs. The emitter still checks the range where the `Const`
    /// narrows to an immediate (`shift_immediate` in `program/lower.rs`), for
    /// a term that reaches it without passing through a graph.
    #[must_use]
    pub fn admits(self, kind: OpKind) -> bool {
        let runtime_extra = self == Self::Runtime
            && matches!(
                kind,
                OpKind::BitAnd
                    | OpKind::BitOr
                    | OpKind::TruncToInt
                    | OpKind::IntToFloat
                    | OpKind::IAdd
                    | OpKind::Shl
                    | OpKind::Shr
                    | OpKind::Bits
                    | OpKind::Float
            );
        // Opaque in both: representable as structure (two gathers hash-cons
        // into one e-class iff buffer identity and all three operands match),
        // never nameable by a template.
        kind == OpKind::Gather || runtime_extra || template_may_name(kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact set of kinds a template may name, pinned: any change to
    /// `template_may_name` or `Vocabulary::admits` fails here and has to be
    /// decided. `OpKind::Tuple` is absent by design, since a node holds a
    /// fixed arity.
    const NAMEABLE: [OpKind; 36] = [
        OpKind::Add,
        OpKind::Sub,
        OpKind::Mul,
        OpKind::Div,
        OpKind::Neg,
        OpKind::Sqrt,
        OpKind::Rsqrt,
        OpKind::Abs,
        OpKind::Min,
        OpKind::Max,
        OpKind::MulAdd,
        OpKind::Recip,
        OpKind::Floor,
        OpKind::Ceil,
        OpKind::Round,
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
        OpKind::Lt,
        OpKind::Le,
        OpKind::Gt,
        OpKind::Ge,
        OpKind::Eq,
        OpKind::Ne,
        OpKind::If,
        OpKind::Dwrt,
    ];

    const RUNTIME_ONLY: [OpKind; 9] = [
        OpKind::TruncToInt,
        OpKind::IntToFloat,
        OpKind::IAdd,
        OpKind::Shl,
        OpKind::Shr,
        OpKind::BitAnd,
        OpKind::BitOr,
        OpKind::Bits,
        OpKind::Float,
    ];

    fn frozen(vocabulary: Vocabulary) -> Vec<OpKind> {
        let mut kinds = NAMEABLE.to_vec();
        kinds.push(OpKind::Gather);
        if vocabulary == Vocabulary::Runtime {
            kinds.extend(RUNTIME_ONLY);
        }
        kinds
    }

    #[test]
    fn a_template_names_what_it_always_named() {
        for kind in OpKind::all() {
            assert_eq!(
                template_may_name(kind),
                NAMEABLE.contains(&kind),
                "{kind:?}"
            );
        }
    }

    #[test]
    fn each_vocabulary_admits_what_it_always_admitted() {
        for vocabulary in [Vocabulary::Templates, Vocabulary::Runtime] {
            let admitted = frozen(vocabulary);
            for kind in OpKind::all() {
                assert_eq!(
                    vocabulary.admits(kind),
                    admitted.contains(&kind),
                    "{vocabulary:?} {kind:?}"
                );
            }
        }
    }

    /// A node holds its operands inline, so an operation without a fixed
    /// arity is not one the graph can hold.
    #[test]
    fn every_admitted_operation_fits_an_inline_node() {
        for vocabulary in [Vocabulary::Templates, Vocabulary::Runtime] {
            for kind in OpKind::all().filter(|&kind| vocabulary.admits(kind)) {
                assert!(
                    (1..=crate::egraph::node::MAX_OPERANDS).contains(&kind.arity()),
                    "{vocabulary:?} admits {kind:?}, arity {}",
                    kind.arity()
                );
            }
        }
    }
}
