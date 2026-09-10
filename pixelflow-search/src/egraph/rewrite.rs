//! Rewrite rule infrastructure.

use alloc::sync::Arc;

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::ops::Op;

use pixelflow_ir::Rooted;
use pixelflow_ir::expr::{ExprBuilder, ExprData, ExprRef};

/// A rewrite RHS pattern, shared.
///
/// [`Rooted`] has no `Debug` impl (it is not a debugging-facing type anywhere
/// else in the codebase), but [`RewriteAction`] derives one for assertion
/// failures and test output — so `Instantiate`'s pattern graph is wrapped
/// rather than forcing a `Debug` impl onto `Rooted` itself, which would be a
/// wider API surface change than this justifies.
#[derive(Clone)]
pub struct TemplatePattern(pub Arc<Rooted<ExprData>>);

pub type TemplateArena = TemplatePattern;

impl core::fmt::Debug for TemplatePattern {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "TemplatePattern(..)")
    }
}

/// Build a rewrite-rule template directly into an [`ExprBuilder`], returning
/// the root [`ExprRef`]. The DSL mirrors the structural pattern: `var N` / `cst V` /
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
        outer: &'static dyn Op,
        inner: &'static dyn Op,
        a: EClassId,
        b: EClassId,
        c: EClassId,
    },
    /// Factor: A*B + A*C -> A * (B + C)
    Factor {
        outer: &'static dyn Op,
        inner: &'static dyn Op,
        common: EClassId,
        unique_l: EClassId,
        unique_r: EClassId,
    },
    /// Canonicalize: Sub(a,b) -> Add(a, Neg(b))
    Canonicalize {
        target: &'static dyn Op,
        inverse: &'static dyn Op,
        a: EClassId,
        b: EClassId,
    },
    /// Associate: (a op b) op c -> a op (b op c)
    Associate {
        op: &'static dyn Op,
        a: EClassId,
        b: EClassId,
        c: EClassId,
    },
    /// ReverseAssociate: a op (b op c) -> (a op b) op c
    ReverseAssociate {
        op: &'static dyn Op,
        a: EClassId,
        b: EClassId,
        c: EClassId,
    },
    /// OddParity: Op(neg(x)) -> neg(Op(x))
    /// Creates Op(inner), then wraps in Neg.
    OddParity {
        func: &'static dyn Op,
        inner: EClassId,
    },
    /// AngleAddition: sin(a+b) -> sin(a)cos(b) + cos(a)sin(b)
    /// or cos(a+b) -> cos(a)cos(b) - sin(a)sin(b)
    AngleAddition {
        term1_op1: &'static dyn Op,
        term1_op2: &'static dyn Op,
        term2_op1: &'static dyn Op,
        term2_op2: &'static dyn Op,
        term2_sign: crate::math::trig::Sign,
        a: EClassId,
        b: EClassId,
    },
    /// Homomorphism: f(a ⊕ b) -> f(a) ⊗ f(b)
    /// e.g., exp(a + b) -> exp(a) * exp(b)
    Homomorphism {
        func: &'static dyn Op,
        target_op: &'static dyn Op,
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
        trig_op: &'static dyn Op,
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
        log_op: &'static dyn Op,
        base: EClassId,
        exponent: EClassId,
    },
    /// ExpandSquare: (a+b)² -> a² + 2ab + b²
    ExpandSquare { a: EClassId, b: EClassId },
    /// DiffOfSquares: a² - b² -> (a+b)(a-b)
    DiffOfSquares { a: EClassId, b: EClassId },
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

    /// Try to apply this rule to a node in an e-class.
    /// Returns `Some(action)` if the rule matches.
    fn apply(&self, egraph: &EGraph, id: EClassId, node: &ENode) -> Option<RewriteAction>;

    /// Whether this rule is destructive: the matched LHS node should be
    /// removed from the e-class after the action is applied.
    ///
    /// Only safe for rules that provably simplify: the RHS is strictly
    /// cheaper than the LHS (involution, identity, annihilator, constant-fold).
    /// Destructive rules reduce e-graph size, preventing node accumulation
    /// that slows future rule matching.
    ///
    /// Default: false (non-destructive, standard equality saturation).
    fn is_destructive(&self) -> bool {
        false
    }

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
    fn lhs_template(&self, _out: &mut ExprBuilder) -> Option<ExprRef> {
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
    fn rhs_template(&self, _out: &mut ExprBuilder) -> Option<ExprRef> {
        None
    }
}
