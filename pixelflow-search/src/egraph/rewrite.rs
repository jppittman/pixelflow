//! Rewrite rule infrastructure.

use alloc::sync::Arc;

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::ops::Vocabulary;
use pixelflow_ir::arena::{ExprArena, ExprId};

use pixelflow_ir::OpKind;
use pixelflow_ir::Rooted;
/// A rewrite RHS pattern, shared.
///
/// [`ExprArena`] has no `Debug` impl (it is not a debugging-facing type
/// anywhere else in the codebase), but [`RewriteAction`] derives one for
/// assertion failures and test output — so `Instantiate`'s pattern arena is
/// wrapped rather than forcing a `Debug` impl onto `ExprArena` itself,
/// which would be a wider API surface change than this justifies.
use pixelflow_ir::expr::ExprData;
use pixelflow_ir::kind::OpSet;

/// A rewrite RHS pattern, shared.
#[derive(Clone)]
pub struct TemplatePattern(pub Arc<Rooted<ExprData>>);

pub type TemplateArena = TemplatePattern;

impl core::fmt::Debug for TemplatePattern {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "TemplatePattern(..)")
    }
}

/// Build a rewrite-rule template directly into an [`ExprArena`], returning the
/// root [`ExprId`]. The DSL mirrors the structural pattern: `var N` / `cst V` /
/// `par N` leaves, and `un OP, (..)` / `bin OP, (..), (..)` / `tern OP, (..),
/// (..), (..)` nodes. Metavariables are encoded as `var N` (Var(0) = A, etc.).
#[macro_export]
macro_rules! arena_pat {
    ($a:expr, var $i:expr) => { $a.push_var($i) };
    ($a:expr, cst $v:expr) => { $a.push_const($v) };
    ($a:expr, par $i:expr) => { $a.push_param($i) };
    ($a:expr, un $op:expr, ($($c:tt)+)) => {{
        let __c = $crate::arena_pat!($a, $($c)+);
        $a.push_unary($op, __c)
    }};
    ($a:expr, bin $op:expr, ($($l:tt)+), ($($r:tt)+)) => {{
        let __l = $crate::arena_pat!($a, $($l)+);
        let __r = $crate::arena_pat!($a, $($r)+);
        $a.push_binary($op, __l, __r)
    }};
    ($a:expr, tern $op:expr, ($($x:tt)+), ($($y:tt)+), ($($z:tt)+)) => {{
        let __x = $crate::arena_pat!($a, $($x)+);
        let __y = $crate::arena_pat!($a, $($y)+);
        let __z = $crate::arena_pat!($a, $($z)+);
        $a.push_ternary($op, __x, __y, __z)
    }};
}

/// Actions that a rewrite rule can produce.
#[derive(Debug, Clone)]
pub enum RewriteAction {
    /// Union this e-class with another
    Union(EClassId),
    /// Create a new e-node and union with it
    Create(ENode),
    /// Distribute: A * (B + C) -> A*B + A*C
    Distribute {
        outer: OpKind,
        inner: OpKind,
        a: EClassId,
        b: EClassId,
        c: EClassId,
    },
    /// Factor: A*B + A*C -> A * (B + C)
    Factor {
        outer: OpKind,
        inner: OpKind,
        common: EClassId,
        unique_l: EClassId,
        unique_r: EClassId,
    },
    /// Canonicalize: Sub(a,b) -> Add(a, Neg(b))
    Canonicalize {
        target: OpKind,
        inverse: OpKind,
        a: EClassId,
        b: EClassId,
    },
    /// Associate: (a op b) op c -> a op (b op c)
    Associate {
        op: OpKind,
        a: EClassId,
        b: EClassId,
        c: EClassId,
    },
    /// ReverseAssociate: a op (b op c) -> (a op b) op c
    ReverseAssociate {
        op: OpKind,
        a: EClassId,
        b: EClassId,
        c: EClassId,
    },
    /// OddParity: Op(neg(x)) -> neg(Op(x))
    /// Creates Op(inner), then wraps in Neg.
    OddParity { func: OpKind, inner: EClassId },
    /// AngleAddition: sin(a+b) -> sin(a)cos(b) + cos(a)sin(b)
    /// or cos(a+b) -> cos(a)cos(b) - sin(a)sin(b)
    AngleAddition {
        term1_op1: OpKind,
        term1_op2: OpKind,
        term2_op1: OpKind,
        term2_op2: OpKind,
        term2_sign: crate::math::trig::Sign,
        a: EClassId,
        b: EClassId,
    },
    /// Homomorphism: f(a ⊕ b) -> f(a) ⊗ f(b)
    /// e.g., exp(a + b) -> exp(a) * exp(b)
    Homomorphism {
        func: OpKind,
        target_op: OpKind,
        a: EClassId,
        b: EClassId,
    },
    /// PowerCombine: x^a * x^b -> x^(a+b)
    PowerCombine {
        base: EClassId,
        exp_a: EClassId,
        exp_b: EClassId,
    },
    /// ReverseAngleAddition: sin(a)cos(b) + cos(a)sin(b) -> sin(a + b)
    /// (The inverse of angle addition, enables double angle discovery)
    ReverseAngleAddition {
        trig_op: OpKind,
        a: EClassId,
        b: EClassId,
    },
    /// HalfAngleProduct: sin(x) * cos(x) -> sin(x + x) / 2
    /// Derived from sin(2x) = 2*sin(x)*cos(x)
    HalfAngleProduct { x: EClassId },
    /// Doubling: a + a -> 2 * a
    Doubling { a: EClassId },
    /// Halving: 2 * a -> a + a (reverse of doubling)
    Halving { a: EClassId },
    /// PowerRecurrence: pow(x, n) -> x * pow(x, n-1) for integer n >= 3
    PowerRecurrence { base: EClassId, exponent: i32 },
    /// LogPower: log(pow(x, n)) -> n * log(x)
    LogPower {
        log_op: OpKind,
        base: EClassId,
        exponent: EClassId,
    },
    /// ExpandSquare: (a+b)² -> a² + 2ab + b²
    ExpandSquare { a: EClassId, b: EClassId },
    /// DiffOfSquares: a² - b² -> (a+b)(a-b)
    DiffOfSquares { a: EClassId, b: EClassId },
    /// Build a right-hand side the rule planned, and union it with the
    /// matched class: every fold rule's action (`PeelFold`, `HalveFold`,
    /// `FactorFold`).
    ///
    /// A plan of nodes carrying their own `OpKind`s rather than an arena
    /// template, because what these rules build is, in part, a copy of a
    /// term the graph already holds — a substituted body — and may contain
    /// any op the graph holds, while the arena-template path instantiates
    /// only what `template_may_name` admits, which deliberately is what a
    /// *rewrite rule* may name and so refuses a `Gather` or a mask. One
    /// variant for all of them, replayed by one function, so predicting a
    /// rule's growth and committing it cannot drift apart per rule. See
    /// [`Plan`](super::fold_rules::Plan).
    ///
    /// A plan can be declined when it is built: a fold of it whose value bound
    /// is a class whose fact varies with a coordinate or with the fold's own
    /// binder (L3 of docs/language/2026-10-20-a-bound-is-a-value.md) is not
    /// added, and the matched class is not unioned with anything. The nodes
    /// built before that fold stay, and the prediction counts them.
    Plan(super::fold_rules::Plan),

    /// Differentiate: expand `Dwrt(inner, var)` one chain-rule step.
    ///
    /// `inner` is a representative node of the expression being differentiated;
    /// `var` is the variable index it is taken with respect to. The executor
    /// builds the derivative subtree, emitting fresh `Dwrt` nodes for any
    /// sub-expressions so equality saturation keeps pushing the derivative
    /// toward the leaves until it dissolves into ordinary arithmetic.
    Differentiate { inner: ENode, var: u8 },

    /// Instantiate `template`'s subtree rooted at `root` bottom-up, reading
    /// each `Var(i)` leaf as `bindings[i]` (an existing e-class — no new
    /// node is created for a metavariable, exactly as a real rewrite
    /// target), building every internal node via `EGraph::add`, then
    /// unioning the result with the class that matched the pattern's LHS.
    ///
    /// # Why this is the only action Round 2 adds
    ///
    /// A mechanical composition `A∘B` has no fixed shape — it is *data* (an
    /// RHS pattern discovered at harness startup, see
    /// [`super::template::TemplateRewrite`]), not a hand-written
    /// combinator. And a hand-written rule whose RHS is already spelled as
    /// an `rhs_template` gains nothing from a bespoke variant that rebuilds
    /// the same shape a second way: two spellings of one shape is exactly
    /// the drift this codebase's one-constructor rule exists to prevent, and
    /// the second spelling is the one no oracle test reads. So
    /// `crate::math::round2_rules` — 33 harness-only rules — produces this
    /// action and nothing else, and this enum grows by one variant rather
    /// than by twelve.
    Instantiate {
        /// The RHS pattern.
        template: TemplatePattern,
        /// Which entry of `template` is the pattern's root.
        entry: usize,
        /// `bindings[i]` is the e-class the pattern's `Var(i)` names.
        bindings: alloc::vec::Vec<EClassId>,
    },
}

/// The node shapes on which a rule's [`Rewrite::apply`] can return `Some`.
///
/// A node's *shape* is its [`OpKind`]: for an [`ENode::Op`] the op's own, for
/// an [`ENode::Shift`] the shift its direction names, and
/// for every other node the kind that names its variant (`Var`, `Const`,
/// `Buffer`, `Uniform`, `Param`, `Reduce`). A [`ENode::Ref`] names no kind,
/// so only [`RootShapes::Any`] admits one.
///
/// The set is a *claim about `apply`*, and the scan spends it: a node whose
/// shape is outside it is never handed to the rule, because the rule would
/// have declined it. A set smaller than the truth therefore silently loses
/// rewrites; a larger one costs only the calls it was meant to save. That
/// asymmetry is why "matches anything" is a named answer, [`RootShapes::Any`],
/// and why [`Rewrite::root_shapes`] has no default: a rule's author states the
/// shapes, and a forgotten impl is a compile error rather than a rule the
/// scan quietly never calls.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RootShapes {
    /// Every node, including a [`ENode::Ref`]: for a rule that genuinely
    /// inspects the root's shape itself, or a probe that must see them all.
    Any,
    /// Exactly the nodes whose shape is marked.
    Kinds(OpSet),
}

// Prints the kinds as a bare set, so a failed assertion names them directly.
impl core::fmt::Debug for RootShapes {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Any => f.write_str("Any"),
            Self::Kinds(kinds) => f.debug_set().entries(kinds.iter()).finish(),
        }
    }
}

impl RootShapes {
    /// No node: a rule that can never match.
    #[must_use]
    pub const fn nothing() -> Self {
        Self::Kinds(OpSet::EMPTY)
    }

    /// Nodes of this one shape.
    #[must_use]
    pub fn kind(kind: OpKind) -> Self {
        Self::kinds(&[kind])
    }

    /// Nodes of any of these shapes.
    #[must_use]
    pub fn kinds(kinds: &[OpKind]) -> Self {
        Self::Kinds(kinds.iter().fold(OpSet::EMPTY, |set, &kind| set.with(kind)))
    }

    /// Every operation node, [`ENode::Op`] whatever its op or [`ENode::Shift`]
    /// by its direction's op: the kinds a runtime-tier graph can hold as an
    /// operation, which [`Vocabulary::Runtime`] already states.
    #[must_use]
    pub fn every_op() -> Self {
        Self::Kinds(
            OpKind::all()
                .filter(|&kind| Vocabulary::Runtime.admits(kind))
                .fold(OpSet::EMPTY, OpSet::with),
        )
    }

    /// Whether `node` is in the set.
    #[must_use]
    pub fn admits(&self, node: &ENode) -> bool {
        match self {
            Self::Any => true,
            Self::Kinds(kinds) => node.shape().is_some_and(|kind| kinds.contains(kind)),
        }
    }
}

/// A rewrite rule that can be applied to e-graph nodes.
///
/// Requires `Send + Sync` so rules can be shared across worker threads
/// during parallel trajectory generation.
pub trait Rewrite: Send + Sync {
    /// Human-readable name for debugging.
    ///
    /// This is the *family* name, and eleven families have more than one
    /// instance in [`crate::egraph::all_rules`] — four `Commutative`s, four
    /// `Associative`s, and so on. A name alone therefore does not identify a
    /// rule; [`specialization`](Rewrite::specialization) is the other half.
    /// Reach for [`crate::egraph::RuleId`] whenever the answer has to survive
    /// a reordering or an insertion.
    fn name(&self) -> &str;

    /// The operator this instance of the family is specialised to, when the
    /// family has more than one instance.
    ///
    /// `Commutative` is instantiated four times — over `Add`, `Mul`, `Min`,
    /// `Max` — and all four answer `"commutative"` to
    /// [`name`](Rewrite::name). The pair `(name, specialization)` is what
    /// tells them apart, and is what [`crate::egraph::RuleId`] hashes.
    ///
    /// Answering with the *operator* rather than a decorated string is the
    /// subtraction: no per-family string table, no arm to forget, and no
    /// fallback that would silently re-alias an op nobody enumerated.
    ///
    /// Default: `None` — correct for every single-instance rule.
    fn specialization(&self) -> Option<pixelflow_ir::OpKind> {
        None
    }

    /// The shapes of root node this rule's [`apply`](Rewrite::apply) can
    /// match: the scan does not call `apply` on a node outside the set.
    ///
    /// Derived from `apply`'s own guards on the node it is handed, not from
    /// the shapes it goes on to read in the node's children. Required, with no
    /// default: see [`RootShapes`].
    fn root_shapes(&self) -> RootShapes;

    /// Try to apply this rule to a node in an e-class.
    /// Returns `Some(action)` if the rule matches.
    fn apply(&self, egraph: &EGraph, id: EClassId, node: &ENode) -> Option<RewriteAction>;

    /// LHS template expression (what this rule matches).
    ///
    /// Uses metavariables: `Expr::Var(0)` = A, `Expr::Var(1)` = B, etc.
    /// These describe the structural pattern that triggers the rule.
    ///
    /// Example: Distribute rule (`A * (B + C) → A*B + A*C`) would return:
    /// ```ignore
    /// Expr::Binary(Mul, Var(0), Expr::Binary(Add, Var(1), Var(2)))
    /// ```
    ///
    /// Returns `None` if the rule doesn't have a defined template.
    /// Rules can opt-in by overriding this method.
    fn lhs_template(&self, _arena: &mut ExprArena) -> Option<ExprId> {
        None
    }

    /// RHS template expression (what this rule produces).
    ///
    /// Uses the same metavariables as `lhs_template()`.
    ///
    /// Example: Distribute rule would return:
    /// ```ignore
    /// Expr::Binary(Add,
    ///     Expr::Binary(Mul, Var(0), Var(1)),
    ///     Expr::Binary(Mul, Var(0), Var(2)))
    /// ```
    ///
    /// Returns `None` if the rule doesn't have a defined template.
    fn rhs_template(&self, _arena: &mut ExprArena) -> Option<ExprId> {
        None
    }
}
