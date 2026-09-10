//! Round 2, mode (iii): genuinely new rewrite rules.
//!
//! `docs/plans/2026-09-01-phase3-round2-rule-scaling.md` §2.3 — the fidelity
//! arm of Phase 3 Round 2 ("does the Guide's advantage grow with the rule
//! count?"). This module is **harness-only**: [`experimental_rules`] is not
//! called by [`super::all_rules`] or [`super::all_math_rules`], and nothing
//! here changes production behavior. Adoption of any rule into production is
//! JP's decision, made after the Round 2 measurement, on its own PR.
//!
//! ## The algebraic-validity contract
//!
//! Every rule below is a theorem over the reals on the *documented domain*
//! of the ops it touches (`docs/plans/2026-08-05-egraph-nnue-research-workflow.md`
//! §0.4; CLAUDE.md "Precision is on the table; range is not"). No domain
//! guard is added to make an identity "safer" than the ops it is built from
//! — divergence outside a documented domain (e.g. `sqrt` of a negative
//! literal) is contract, not a soundness gap, and is exercised only at
//! well-conditioned points by this module's oracle tests. Constant-guarded
//! rules (N7, N28) carry a side condition on a *literal* — a matching
//! condition, never an IEEE guard.
//!
//! Two checks, per §2.4, never conflated:
//! - **Same-form hard gate**: `pixelflow_ir::eval_scalar` vs the JIT on the
//!   same arena — pixelflow-ir's existing parity suite. Rewrite rules never
//!   touch lowering, so this module does not re-run it; it is not what the
//!   tests below check.
//! - **Cross-form conditioned gate**: this module's tests. Each rule's
//!   `lhs_template`/`rhs_template` share the same metavariable encoding
//!   `eval_scalar` already understands (`Var(i)`), so "instantiate LHS and
//!   RHS with the same leaf assignment" is exactly "evaluate both templates
//!   at the same point" — see `round2_oracle_check` below.
//!
//! ## Naming
//!
//! Struct names follow the design doc's family numbering in doc comments
//! (`N1`..`N28`); the Rust identifiers are descriptive, not `N`-numbered,
//! matching the rest of this crate's style (`FmaFusion`, not `Rule27`).

use alloc::sync::Arc;

use crate::arena_pat;
use crate::egraph::{EClassId, EGraph, ENode, Rewrite, RewriteAction, TemplatePattern, ops};
use core::f32::consts::{LN_2, LOG2_E, LOG10_2};
use pixelflow_ir::OpKind;
use pixelflow_ir::expr::{ExprBuilder, ExprRef};

// ============================================================================
// Shared helpers (module-local; see fusion.rs / power.rs for the established
// per-module extract_* convention — no shared crate-level helper exists for
// this today, and this module keeps its own for the same reason those do).
// ============================================================================

/// This rule's own RHS pattern, instantiated under `bindings`.
///
/// Every rule in this module already spells its RHS exactly once, as the
/// `rhs_template` the cross-form oracle test at the bottom of this file
/// reads. Re-spelling the same shape a second time as a bespoke
/// Instantiate a rule's RHS via its `rhs_template()`.
///
/// Every rule in this module has a fixed-shape RHS, and its `apply` method
/// produces a `RewriteAction::Instantiate` pointing at that shape rather
/// than handwriting its own node-building loop.
///
/// # Panics
///
/// If `rule` has no `rhs_template`.
fn instantiate_rhs(rule: &dyn Rewrite, bindings: Vec<EClassId>) -> RewriteAction {
    let mut arena = ExprBuilder::new();
    let root = rule.rhs_template(&mut arena).unwrap_or_else(|| {
        panic!(
            "round2_rules: {} has no rhs_template — every rule in this module \
             instantiates its own RHS",
            rule.name()
        )
    });
    let (rooted, _env) = arena.finish(&[root]);
    RewriteAction::Instantiate {
        template: TemplatePattern(Arc::new(rooted)),
        entry: 0,
        bindings,
    }
}

/// Extract the constant value held by an e-class, if any node in it is one.
fn eclass_const(egraph: &EGraph, id: EClassId) -> Option<f32> {
    egraph.nodes(id).iter().find_map(ENode::as_f32)
}

/// If any node in `class` is `Op(kind)` with exactly one child, return it.
fn extract_unary(egraph: &EGraph, class: EClassId, kind: OpKind) -> Option<EClassId> {
    for node in egraph.nodes(class) {
        if let ENode::Op { op, children } = node {
            if op.kind() == kind && children.len() == 1 {
                return Some(children[0]);
            }
        }
    }
    None
}

/// If any node in `class` is `Op(kind)` with exactly two children, return them.
fn extract_binary(egraph: &EGraph, class: EClassId, kind: OpKind) -> Option<(EClassId, EClassId)> {
    for node in egraph.nodes(class) {
        if let ENode::Op { op, children } = node {
            if op.kind() == kind && children.len() == 2 {
                return Some((children[0], children[1]));
            }
        }
    }
    None
}

// ============================================================================
// N1 / N2 — min/max De Morgan duality
// ============================================================================

/// `min(a,b) = neg(max(neg a, neg b))`, and its dual
/// `max(a,b) = neg(min(neg a, neg b))` (N1, N2).
///
/// A theorem over the reals. The platform-specific NaN/signed-zero rows for
/// `Min`/`Max` (CLAUDE.md's floating-point table) diverge between backends by
/// design; this identity is exercised only at well-conditioned (non-NaN,
/// non-zero) points, matching the contract.
struct MinMaxDuality {
    from: OpKind,
    dual: OpKind,
    name: &'static str,
}

impl Rewrite for MinMaxDuality {
    fn name(&self) -> &str {
        self.name
    }

    fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != self.from || children.len() != 2 {
            return None;
        }
        Some(instantiate_rhs(self, vec![children[0], children[1]]))
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin self.from, (var 0), (var 1)))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, un OpKind::Neg,
            (bin self.dual, (un OpKind::Neg, (var 0)), (un OpKind::Neg, (var 1)))))
    }
}

fn min_max_duality_rules() -> Vec<Box<dyn Rewrite>> {
    vec![
        Box::new(MinMaxDuality {
            from: OpKind::Min,
            dual: OpKind::Max,
            name: "min-max-duality-min",
        }),
        Box::new(MinMaxDuality {
            from: OpKind::Max,
            dual: OpKind::Min,
            name: "min-max-duality-max",
        }),
    ]
}

// ============================================================================
// N3 / N4 — min/max absorption
// ============================================================================

/// `min(a, max(a,b)) = a`, and its dual `max(a, min(a,b)) = a` (N3, N4).
/// Lattice absorption, a theorem over the reals.
struct MinMaxAbsorption {
    outer: OpKind,
    inner: OpKind,
    name: &'static str,
}

impl Rewrite for MinMaxAbsorption {
    fn name(&self) -> &str {
        self.name
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != self.outer || children.len() != 2 {
            return None;
        }
        for (candidate_a, other) in [(children[0], children[1]), (children[1], children[0])] {
            if let Some((ia, ib)) = extract_binary(egraph, other, self.inner) {
                let a_canon = egraph.find(candidate_a);
                if egraph.find(ia) == a_canon || egraph.find(ib) == a_canon {
                    return Some(RewriteAction::Union(candidate_a));
                }
            }
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin self.outer, (var 0), (bin self.inner, (var 0), (var 1))))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, var 0))
    }
}

fn min_max_absorption_rules() -> Vec<Box<dyn Rewrite>> {
    vec![
        Box::new(MinMaxAbsorption {
            outer: OpKind::Min,
            inner: OpKind::Max,
            name: "min-max-absorption-min",
        }),
        Box::new(MinMaxAbsorption {
            outer: OpKind::Max,
            inner: OpKind::Min,
            name: "min-max-absorption-max",
        }),
    ]
}

// ============================================================================
// N5 / N6 — translation distributes over min/max
// ============================================================================

/// `minmax(a,b) + c = minmax(a+c, b+c)` (N5 for min, N6 for max).
/// Translation is monotone, so it commutes with min/max over the reals.
struct MinMaxTranslate {
    minmax: OpKind,
    name: &'static str,
}

impl Rewrite for MinMaxTranslate {
    fn name(&self) -> &str {
        self.name
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Add || children.len() != 2 {
            return None;
        }
        for (inner_cand, c) in [(children[0], children[1]), (children[1], children[0])] {
            if let Some((a, b)) = extract_binary(egraph, inner_cand, self.minmax) {
                let minmax_op = ops::op_from_kind(self.minmax)
                    .unwrap_or_else(|| panic!("MinMaxTranslate: {:?} has no Op", self.minmax));
                // Distribute{outer=Add, inner=minmax, a=c, b=a, c=b} builds
                // minmax(Add(c,a), Add(c,b)) = minmax(a+c, b+c) up to Add's
                // commutativity (which the e-graph already carries).
                return Some(RewriteAction::Distribute {
                    outer: &ops::Add,
                    inner: minmax_op,
                    a: c,
                    b: a,
                    c: b,
                });
            }
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Add, (bin self.minmax, (var 0), (var 1)), (var 2)))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin self.minmax,
            (bin OpKind::Add, (var 0), (var 2)), (bin OpKind::Add, (var 1), (var 2))))
    }
}

fn min_max_translate_rules() -> Vec<Box<dyn Rewrite>> {
    vec![
        Box::new(MinMaxTranslate {
            minmax: OpKind::Min,
            name: "min-translate",
        }),
        Box::new(MinMaxTranslate {
            minmax: OpKind::Max,
            name: "max-translate",
        }),
    ]
}

// ============================================================================
// N7 — nonneg-literal scaling distributes over min
// ============================================================================

/// `k * min(a,b) = min(k*a, k*b)` for a literal `k >= 0` (N7).
///
/// Constant-guarded: the side condition is on the *literal* `k`'s sign, a
/// matching condition per the contract, not an IEEE guard. For negative `k`
/// the identity is `k*min(a,b) = max(k*a,k*b)` instead — this rule simply
/// does not fire there (a stricter match, not a soundness gap).
struct MinScaledByNonnegLiteral;

impl Rewrite for MinScaledByNonnegLiteral {
    fn name(&self) -> &str {
        "min-scaled-by-nonneg-literal"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Mul || children.len() != 2 {
            return None;
        }
        for (k_cand, other) in [(children[0], children[1]), (children[1], children[0])] {
            let Some(k) = eclass_const(egraph, k_cand) else {
                continue;
            };
            if k < 0.0 {
                continue;
            }
            if let Some((a, b)) = extract_binary(egraph, other, OpKind::Min) {
                return Some(RewriteAction::Distribute {
                    outer: &ops::Mul,
                    inner: &ops::Min,
                    a: k_cand,
                    b: a,
                    c: b,
                });
            }
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Mul, (cst 2.0), (bin OpKind::Min, (var 0), (var 1))))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Min,
            (bin OpKind::Mul, (cst 2.0), (var 0)), (bin OpKind::Mul, (cst 2.0), (var 1))))
    }
}

// ============================================================================
// N8 — min/max lattice distributivity
// ============================================================================

/// `min(a, max(b,c)) = max(min(a,b), min(a,c))` (N8). Lattice
/// distributivity, a theorem over the reals.
struct MinMaxDistributive;

impl Rewrite for MinMaxDistributive {
    fn name(&self) -> &str {
        "min-max-distributive"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Min || children.len() != 2 {
            return None;
        }
        for (a, other) in [(children[0], children[1]), (children[1], children[0])] {
            if let Some((b, c)) = extract_binary(egraph, other, OpKind::Max) {
                return Some(RewriteAction::Distribute {
                    outer: &ops::Min,
                    inner: &ops::Max,
                    a,
                    b,
                    c,
                });
            }
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Min, (var 0), (bin OpKind::Max, (var 1), (var 2))))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Max,
            (bin OpKind::Min, (var 0), (var 1)), (bin OpKind::Min, (var 0), (var 2))))
    }
}

// ============================================================================
// N9 / N10 — abs as a lattice op
// ============================================================================

/// `abs(x) = max(x, neg(x))` (N9). Abs as a lattice op, a theorem over the
/// reals.
struct AbsAsMax;

impl Rewrite for AbsAsMax {
    fn name(&self) -> &str {
        "abs-as-max"
    }

    fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Abs || children.len() != 1 {
            return None;
        }
        Some(instantiate_rhs(self, vec![children[0]]))
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, un OpKind::Abs, (var 0)))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Max, (var 0), (un OpKind::Neg, (var 0))))
    }
}

/// `max(x, neg(x)) = abs(x)` (N10). The fusion direction, reverse of N9.
struct MaxSelfNegAsAbs;

impl Rewrite for MaxSelfNegAsAbs {
    fn name(&self) -> &str {
        "max-self-neg-as-abs"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Max || children.len() != 2 {
            return None;
        }
        for (x, other) in [(children[0], children[1]), (children[1], children[0])] {
            if let Some(negated) = extract_unary(egraph, other, OpKind::Neg) {
                if egraph.find(negated) == egraph.find(x) {
                    return Some(RewriteAction::Create(ENode::Op {
                        op: &ops::Abs,
                        children: vec![x],
                    }));
                }
            }
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Max, (var 0), (un OpKind::Neg, (var 0))))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, un OpKind::Abs, (var 0)))
    }
}

// ============================================================================
// N11 — select is mask-independent when both branches agree
// ============================================================================

/// `select(m, a, a) = a` (N11). True for any mask value — a bitwise blend of
/// a value with itself is that value, on every backend.
struct SelectSameBranch;

impl Rewrite for SelectSameBranch {
    fn name(&self) -> &str {
        "select-same-branch"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Select || children.len() != 3 {
            return None;
        }
        let (a, b) = (children[1], children[2]);
        if egraph.find(a) == egraph.find(b) {
            return Some(RewriteAction::Union(a));
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, tern OpKind::Select, (var 0), (var 1), (var 1)))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, var 1))
    }
}

// ============================================================================
// N12 / N13 — select of a comparison is min/max
// ============================================================================

/// `select(lt(a,b), a, b) = min(a,b)` (N12). Over the reals; the platform
/// NaN rows for `Lt` and `Select` diverge by design (CLAUDE.md), so no value
/// is promised there and this rule is exercised only at well-conditioned
/// (non-NaN) points.
struct SelectLtToMin;

impl Rewrite for SelectLtToMin {
    fn name(&self) -> &str {
        "select-lt-to-min"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Select || children.len() != 3 {
            return None;
        }
        let (m, x, y) = (children[0], children[1], children[2]);
        let (a, b) = extract_binary(egraph, m, OpKind::Lt)?;
        if egraph.find(a) == egraph.find(x) && egraph.find(b) == egraph.find(y) {
            return Some(RewriteAction::Create(ENode::Op {
                op: &ops::Min,
                children: vec![x, y],
            }));
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(
            arena_pat!(__a, tern OpKind::Select, (bin OpKind::Lt, (var 0), (var 1)), (var 0), (var 1)),
        )
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Min, (var 0), (var 1)))
    }
}

/// `select(lt(a,b), b, a) = max(a,b)` (N13). As N12.
struct SelectLtToMax;

impl Rewrite for SelectLtToMax {
    fn name(&self) -> &str {
        "select-lt-to-max"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Select || children.len() != 3 {
            return None;
        }
        let (m, x, y) = (children[0], children[1], children[2]);
        let (a, b) = extract_binary(egraph, m, OpKind::Lt)?;
        // select(lt(a,b), b, a): x should be b, y should be a.
        if egraph.find(a) == egraph.find(y) && egraph.find(b) == egraph.find(x) {
            return Some(RewriteAction::Create(ENode::Op {
                op: &ops::Max,
                children: vec![x, y],
            }));
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(
            arena_pat!(__a, tern OpKind::Select, (bin OpKind::Lt, (var 0), (var 1)), (var 1), (var 0)),
        )
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Max, (var 0), (var 1)))
    }
}

// ============================================================================
// N14 — hoist a unary op through select
// ============================================================================

/// `select(m, f(a), f(b)) = f(select(m, a, b))` for unary `f` (N14a–c: Neg,
/// Abs, Sqrt). One `f` instead of two; `select` is a per-lane bitwise blend
/// of one full operand, so applying `f` before or after the blend agrees
/// exactly for any mask value.
struct SelectHoistUnary {
    func: OpKind,
    name: &'static str,
}

impl Rewrite for SelectHoistUnary {
    fn name(&self) -> &str {
        self.name
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Select || children.len() != 3 {
            return None;
        }
        let (m, x, y) = (children[0], children[1], children[2]);
        let a = extract_unary(egraph, x, self.func)?;
        let b = extract_unary(egraph, y, self.func)?;
        Some(instantiate_rhs(self, vec![m, a, b]))
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(
            arena_pat!(__a, tern OpKind::Select, (var 0), (un self.func, (var 1)), (un self.func, (var 2))),
        )
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, un self.func, (tern OpKind::Select, (var 0), (var 1), (var 2))))
    }
}

fn select_hoist_unary_rules() -> Vec<Box<dyn Rewrite>> {
    vec![
        Box::new(SelectHoistUnary {
            func: OpKind::Neg,
            name: "select-hoist-neg",
        }),
        Box::new(SelectHoistUnary {
            func: OpKind::Abs,
            name: "select-hoist-abs",
        }),
        Box::new(SelectHoistUnary {
            func: OpKind::Sqrt,
            name: "select-hoist-sqrt",
        }),
    ]
}

// ============================================================================
// N15 — comparison flip
// ============================================================================

/// `lt(a,b) = gt(b,a)` (N15). Reordering the same relation; a mask-domain
/// result on both sides (CLAUDE.md: a mask is a bit pattern, not a number —
/// this rule never treats one as `1.0`/`0.0`).
struct CompareFlipLt;

impl Rewrite for CompareFlipLt {
    fn name(&self) -> &str {
        "compare-flip-lt"
    }

    fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Lt || children.len() != 2 {
            return None;
        }
        Some(RewriteAction::Create(ENode::Op {
            op: &ops::Gt,
            children: vec![children[1], children[0]],
        }))
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Lt, (var 0), (var 1)))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Gt, (var 1), (var 0)))
    }
}

// ============================================================================
// N16 / N17 — tan definition
// ============================================================================

/// `tan(x) = sin(x) * recip(cos(x))` (N16). The trig-domain restriction on
/// `tan`/`sin`/`cos` (CLAUDE.md: `|x| < TRIG_DOMAIN`) applies unchanged; this
/// rule adds no new domain restriction beyond the ops it is built from.
struct TanDefinition;

impl Rewrite for TanDefinition {
    fn name(&self) -> &str {
        "tan-definition"
    }

    fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Tan || children.len() != 1 {
            return None;
        }
        Some(instantiate_rhs(self, vec![children[0]]))
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, un OpKind::Tan, (var 0)))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Mul,
            (un OpKind::Sin, (var 0)), (un OpKind::Recip, (un OpKind::Cos, (var 0)))))
    }
}

/// `sin(x) * recip(cos(x)) = tan(x)` (N17). Reverse of N16.
struct TanFusion;

impl Rewrite for TanFusion {
    fn name(&self) -> &str {
        "tan-fusion"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Mul || children.len() != 2 {
            return None;
        }
        for (sin_side, recip_side) in [(children[0], children[1]), (children[1], children[0])] {
            let Some(x1) = extract_unary(egraph, sin_side, OpKind::Sin) else {
                continue;
            };
            let Some(cos_class) = extract_unary(egraph, recip_side, OpKind::Recip) else {
                continue;
            };
            let Some(x2) = extract_unary(egraph, cos_class, OpKind::Cos) else {
                continue;
            };
            if egraph.find(x1) == egraph.find(x2) {
                return Some(RewriteAction::Create(ENode::Op {
                    op: &ops::Tan,
                    children: vec![x1],
                }));
            }
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Mul,
            (un OpKind::Sin, (var 0)), (un OpKind::Recip, (un OpKind::Cos, (var 0)))))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, un OpKind::Tan, (var 0)))
    }
}

// ============================================================================
// N18 / N19 / N20 — hardware-native transcendental bases
// ============================================================================

/// `exp(x) = exp2(x * log2(e))` (N18). Hardware-native base; `log2(e)` is
/// `core::f32::consts::LOG2_E`, the same constant either implementation
/// would use, so this is a genuine identity and not a restated definition
/// (CLAUDE.md: "one definition, imported, not restated" — the sibling `ln`,
/// `log10` rules below reuse the same pattern with their own single
/// constant, not a hand-rederived one).
struct ExpAsExp2;

impl Rewrite for ExpAsExp2 {
    fn name(&self) -> &str {
        "exp-as-exp2"
    }

    fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Exp || children.len() != 1 {
            return None;
        }
        Some(instantiate_rhs(self, vec![children[0]]))
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, un OpKind::Exp, (var 0)))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, un OpKind::Exp2, (bin OpKind::Mul, (var 0), (cst LOG2_E))))
    }
}

/// `ln(x) = log2(x) * ln(2)` (N19). Domain `x > 0`.
struct LnAsLog2;

impl Rewrite for LnAsLog2 {
    fn name(&self) -> &str {
        "ln-as-log2"
    }

    fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Ln || children.len() != 1 {
            return None;
        }
        Some(instantiate_rhs(self, vec![children[0]]))
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, un OpKind::Ln, (var 0)))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Mul, (un OpKind::Log2, (var 0)), (cst LN_2)))
    }
}

/// `log10(x) = log2(x) * log10(2)` (N20). Domain `x > 0`.
struct Log10AsLog2;

impl Rewrite for Log10AsLog2 {
    fn name(&self) -> &str {
        "log10-as-log2"
    }

    fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Log10 || children.len() != 1 {
            return None;
        }
        Some(instantiate_rhs(self, vec![children[0]]))
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, un OpKind::Log10, (var 0)))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Mul, (un OpKind::Log2, (var 0)), (cst LOG10_2)))
    }
}

// ============================================================================
// N21 — sqrt product
// ============================================================================

/// `sqrt(x) * sqrt(y) = sqrt(x*y)` (N21). Domain `x, y >= 0`: `eval_scalar`'s
/// `Sqrt` is exact IEEE `sqrtf` (NaN below zero), so for negative operands
/// the two sides diverge (e.g. `x=y=-4`: LHS is `NaN*NaN=NaN`, RHS is
/// `sqrt(16)=4`) — documented domain, not tested outside it.
struct SqrtProduct;

impl Rewrite for SqrtProduct {
    fn name(&self) -> &str {
        "sqrt-product"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Mul || children.len() != 2 {
            return None;
        }
        let x = extract_unary(egraph, children[0], OpKind::Sqrt)?;
        let y = extract_unary(egraph, children[1], OpKind::Sqrt)?;
        Some(instantiate_rhs(self, vec![x, y]))
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(
            arena_pat!(__a, bin OpKind::Mul, (un OpKind::Sqrt, (var 0)), (un OpKind::Sqrt, (var 1))),
        )
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, un OpKind::Sqrt, (bin OpKind::Mul, (var 0), (var 1))))
    }
}

// ============================================================================
// N22 / N23 — rsqrt/sqrt normalize shapes
// ============================================================================

/// `rsqrt(x) * rsqrt(x) = recip(x)` (N22). Domain `x > 0`. `Rsqrt` is a
/// hardware *estimate* (CLAUDE.md), never exact, so this identity is
/// exercised with a loose tolerance, not bit-exactness.
struct RsqrtSquareAsRecip;

impl Rewrite for RsqrtSquareAsRecip {
    fn name(&self) -> &str {
        "rsqrt-square-as-recip"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Mul || children.len() != 2 {
            return None;
        }
        let a = extract_unary(egraph, children[0], OpKind::Rsqrt)?;
        let b = extract_unary(egraph, children[1], OpKind::Rsqrt)?;
        if egraph.find(a) != egraph.find(b) {
            return None;
        }
        Some(RewriteAction::Create(ENode::Op {
            op: &ops::Recip,
            children: vec![a],
        }))
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(
            arena_pat!(__a, bin OpKind::Mul, (un OpKind::Rsqrt, (var 0)), (un OpKind::Rsqrt, (var 0))),
        )
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, un OpKind::Recip, (var 0)))
    }
}

/// `x * rsqrt(x) = sqrt(x)` (N23). Domain `x > 0` (at `x=0`, `rsqrt(0)=inf`
/// and `0*inf=NaN` while `sqrt(0)=0` — the two sides diverge exactly at the
/// boundary CLAUDE.md's `sqrt_fast` discussion calls out).
struct NormalizeAsSqrt;

impl Rewrite for NormalizeAsSqrt {
    fn name(&self) -> &str {
        "normalize-as-sqrt"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Mul || children.len() != 2 {
            return None;
        }
        for (x_cand, rsqrt_side) in [(children[0], children[1]), (children[1], children[0])] {
            if let Some(x2) = extract_unary(egraph, rsqrt_side, OpKind::Rsqrt) {
                if egraph.find(x_cand) == egraph.find(x2) {
                    return Some(RewriteAction::Create(ENode::Op {
                        op: &ops::Sqrt,
                        children: vec![x_cand],
                    }));
                }
            }
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Mul, (var 0), (un OpKind::Rsqrt, (var 0))))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, un OpKind::Sqrt, (var 0)))
    }
}

// ============================================================================
// N24 / N24r — reciprocal product
// ============================================================================

/// `recip(a) * recip(b) = recip(a*b)` (N24). One estimate instead of two.
/// Domain `a, b != 0`; `Recip` is an estimate, loose tolerance.
struct RecipProduct;

impl Rewrite for RecipProduct {
    fn name(&self) -> &str {
        "recip-product"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Mul || children.len() != 2 {
            return None;
        }
        let x = extract_unary(egraph, children[0], OpKind::Recip)?;
        let y = extract_unary(egraph, children[1], OpKind::Recip)?;
        Some(instantiate_rhs(self, vec![x, y]))
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(
            arena_pat!(__a, bin OpKind::Mul, (un OpKind::Recip, (var 0)), (un OpKind::Recip, (var 1))),
        )
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, un OpKind::Recip, (bin OpKind::Mul, (var 0), (var 1))))
    }
}

/// `recip(a*b) = recip(a) * recip(b)` (N24r). Reverse of N24, registered
/// separately per §2.3's note (both directions are useful search moves: one
/// merges two estimates into one, the other exposes the factors to other
/// rules).
struct RecipOfProduct;

impl Rewrite for RecipOfProduct {
    fn name(&self) -> &str {
        "recip-of-product"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Recip || children.len() != 1 {
            return None;
        }
        let (a, b) = extract_binary(egraph, children[0], OpKind::Mul)?;
        Some(instantiate_rhs(self, vec![a, b]))
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, un OpKind::Recip, (bin OpKind::Mul, (var 0), (var 1))))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(
            arena_pat!(__a, bin OpKind::Mul, (un OpKind::Recip, (var 0)), (un OpKind::Recip, (var 1))),
        )
    }
}

// ============================================================================
// N25 / N26 — fused multiply-add: un-fuse and identities
// ============================================================================

/// `fma(a,b,c) = a*b + c` (N25). Un-fuse: re-exposes the sum to other rules
/// (the exact reverse of `fusion::FmaFusion`; both directions coexist so
/// saturation can move either way, unlike production where only the fusion
/// direction is registered).
struct FmaUnfuse;

impl Rewrite for FmaUnfuse {
    fn name(&self) -> &str {
        "fma-unfuse"
    }

    fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::MulAdd || children.len() != 3 {
            return None;
        }
        Some(instantiate_rhs(
            self,
            vec![children[0], children[1], children[2]],
        ))
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, tern OpKind::MulAdd, (var 0), (var 1), (var 2)))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Add, (bin OpKind::Mul, (var 0), (var 1)), (var 2)))
    }
}

/// `fma(a, 1, c) = a + c` (N26, first identity on the fused op).
struct FmaMulIdentity;

impl Rewrite for FmaMulIdentity {
    fn name(&self) -> &str {
        "fma-mul-identity"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::MulAdd || children.len() != 3 {
            return None;
        }
        let b_val = eclass_const(egraph, children[1])?;
        if b_val != 1.0 {
            return None;
        }
        Some(RewriteAction::Create(ENode::Op {
            op: &ops::Add,
            children: vec![children[0], children[2]],
        }))
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, tern OpKind::MulAdd, (var 0), (cst 1.0), (var 1)))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Add, (var 0), (var 1)))
    }
}

/// `fma(a, b, 0) = a * b` (N26, second identity on the fused op).
struct FmaAddIdentity;

impl Rewrite for FmaAddIdentity {
    fn name(&self) -> &str {
        "fma-add-identity"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::MulAdd || children.len() != 3 {
            return None;
        }
        let c_val = eclass_const(egraph, children[2])?;
        if c_val != 0.0 {
            return None;
        }
        Some(RewriteAction::Create(ENode::Op {
            op: &ops::Mul,
            children: vec![children[0], children[1]],
        }))
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, tern OpKind::MulAdd, (var 0), (var 1), (cst 0.0)))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Mul, (var 0), (var 1)))
    }
}

// ============================================================================
// N27 — negation pushes through add/mul
// ============================================================================

/// `neg(a + b) = neg(a) + neg(b)` (N27, first index).
struct NegDistributesAdd;

impl Rewrite for NegDistributesAdd {
    fn name(&self) -> &str {
        "neg-distributes-add"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Neg || children.len() != 1 {
            return None;
        }
        let (a, b) = extract_binary(egraph, children[0], OpKind::Add)?;
        Some(instantiate_rhs(self, vec![a, b]))
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, un OpKind::Neg, (bin OpKind::Add, (var 0), (var 1))))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Add, (un OpKind::Neg, (var 0)), (un OpKind::Neg, (var 1))))
    }
}

/// `neg(a * b) = neg(a) * b` (N27, second index).
struct NegDistributesMul;

impl Rewrite for NegDistributesMul {
    fn name(&self) -> &str {
        "neg-distributes-mul"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Neg || children.len() != 1 {
            return None;
        }
        let (a, b) = extract_binary(egraph, children[0], OpKind::Mul)?;
        Some(instantiate_rhs(self, vec![a, b]))
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, un OpKind::Neg, (bin OpKind::Mul, (var 0), (var 1))))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Mul, (un OpKind::Neg, (var 0)), (var 1)))
    }
}

// ============================================================================
// N28 — division by an exact literal
// ============================================================================

/// `a / k = a * (1/k)` for a literal `k != 0`, with `1/k` computed exactly at
/// rule-application time in `f32` (N28). Constant-guarded: the side
/// condition is on the literal `k` (never zero), not an IEEE guard.
/// `Recip` itself (the op) is never folded — this is not that; it replaces
/// the *division* by an exact literal reciprocal, which is algebraically
/// valid and precision-changing within the contract (CLAUDE.md's "precision
/// is on the table" — a division and a multiply-by-exact-reciprocal are the
/// same operation count with different rounding, which the contract already
/// treats as fair game, not a soundness question).
struct DivByLiteral;

impl Rewrite for DivByLiteral {
    fn name(&self) -> &str {
        "div-by-literal"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Div || children.len() != 2 {
            return None;
        }
        let k = eclass_const(egraph, children[1])?;
        if k == 0.0 {
            return None;
        }
        // The one rule in this module whose RHS is not a fixed shape: the
        // constant is `1/k` for the literal `k` this match found, so the
        // pattern is built here. `rhs_template` below still spells the
        // shape, with a representative literal, for the oracle test.
        let mut arena = ExprBuilder::new();
        let root = arena_pat!(arena, bin OpKind::Mul, (var 0), (cst 1.0 / k));
        let (rooted, _env) = arena.finish(&[root]);
        Some(RewriteAction::Instantiate {
            template: TemplatePattern(Arc::new(rooted)),
            entry: 0,
            bindings: vec![children[0]],
        })
    }

    fn lhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Div, (var 0), (cst 2.0)))
    }

    fn rhs_template(&self, __a: &mut ExprBuilder) -> Option<ExprRef> {
        Some(arena_pat!(__a, bin OpKind::Mul, (var 0), (cst 0.5)))
    }
}

// ============================================================================
// Rule collection
// ============================================================================

/// The Round 2 mode-(iii) batch: genuinely new rules the production 62 lack
/// (`docs/plans/2026-09-01-phase3-round2-rule-scaling.md` §2.3).
///
/// **Harness-only.** Not called by [`super::all_rules`] or
/// [`super::all_math_rules`]; nothing in production behavior changes by this
/// existing. Every rule here passes the cross-form oracle test at the bottom
/// of this file — none were dropped, so this batch is complete against the
/// design's inventory (33 rule indices across 24 families: the design's
/// text estimate of "31" undercounted N24r, which the design also lists as
/// a registered index; the family count and every N-numbered identity match
/// exactly).
pub fn experimental_rules() -> Vec<Box<dyn Rewrite>> {
    let mut rules: Vec<Box<dyn Rewrite>> = Vec::new();
    rules.extend(min_max_duality_rules()); // N1, N2
    rules.extend(min_max_absorption_rules()); // N3, N4
    rules.extend(min_max_translate_rules()); // N5, N6
    rules.push(Box::new(MinScaledByNonnegLiteral)); // N7
    rules.push(Box::new(MinMaxDistributive)); // N8
    rules.push(Box::new(AbsAsMax)); // N9
    rules.push(Box::new(MaxSelfNegAsAbs)); // N10
    rules.push(Box::new(SelectSameBranch)); // N11
    rules.push(Box::new(SelectLtToMin)); // N12
    rules.push(Box::new(SelectLtToMax)); // N13
    rules.extend(select_hoist_unary_rules()); // N14a, N14b, N14c
    rules.push(Box::new(CompareFlipLt)); // N15
    rules.push(Box::new(TanDefinition)); // N16
    rules.push(Box::new(TanFusion)); // N17
    rules.push(Box::new(ExpAsExp2)); // N18
    rules.push(Box::new(LnAsLog2)); // N19
    rules.push(Box::new(Log10AsLog2)); // N20
    rules.push(Box::new(SqrtProduct)); // N21
    rules.push(Box::new(RsqrtSquareAsRecip)); // N22
    rules.push(Box::new(NormalizeAsSqrt)); // N23
    rules.push(Box::new(RecipProduct)); // N24
    rules.push(Box::new(RecipOfProduct)); // N24r
    rules.push(Box::new(FmaUnfuse)); // N25
    rules.push(Box::new(FmaMulIdentity)); // N26a
    rules.push(Box::new(FmaAddIdentity)); // N26b
    rules.push(Box::new(NegDistributesAdd)); // N27a
    rules.push(Box::new(NegDistributesMul)); // N27b
    rules.push(Box::new(DivByLiteral)); // N28
    rules
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::Uniform;
    use pixelflow_ir::binding::BindingTable;
    use pixelflow_ir::eval_scalar;
    use pixelflow_ir::expr::Term;

    /// Well-conditioned sample points, seeded and fixed: finite, moderate
    /// magnitude, no zeros (avoids `Recip`/`Rsqrt` poles), all-positive
    /// available separately for domain-restricted rules (`sqrt`, `ln`,
    /// `log10`, `rsqrt`, N23).
    fn general_points() -> Vec<[f32; 4]> {
        vec![
            [1.3, 0.7, 2.1, -0.9],
            [-2.4, 3.1, -0.5, 1.8],
            [0.6, -1.7, 4.2, 2.9],
            [5.5, -3.3, 1.1, -4.4],
            [0.25, 0.75, -1.25, 3.75],
        ]
    }

    /// All-positive points: for rules whose domain excludes zero/negative
    /// (`sqrt`, `ln`, `log10`, `rsqrt`, `recip`-of-min/max compositions).
    fn positive_points() -> Vec<[f32; 4]> {
        vec![
            [1.3, 0.7, 2.1, 0.9],
            [2.4, 3.1, 0.5, 1.8],
            [0.6, 1.7, 4.2, 2.9],
            [5.5, 3.3, 1.1, 4.4],
            [0.25, 0.75, 1.25, 3.75],
        ]
    }

    /// Points for rules whose `Var(0)` metavariable names a `select` mask
    /// (N14): a mask is a bit pattern, never a number (CLAUDE.md), so
    /// `Var(0)` must be bound to a genuine all-ones/all-zero pattern, not an
    /// arbitrary float — using an arbitrary float there is off-label (no
    /// rule in the language ever constructs `Select` with a non-mask first
    /// argument) and produces meaningless partial-bit blends on both sides
    /// alike, which is a test-harness bug, not a rule counterexample. Other
    /// slots are well-conditioned positive floats (N14c hoists `Sqrt`).
    fn select_mask_points() -> Vec<[f32; 4]> {
        let t = f32::from_bits(u32::MAX);
        let f = 0.0_f32;
        vec![
            [t, 1.3, 2.7, 4.1],
            [f, 1.3, 2.7, 4.1],
            [t, 4.2, 0.5, 1.9],
            [f, 4.2, 0.5, 1.9],
        ]
    }

    /// Replace every metavariable past the coordinate axes with one of
    /// `args`, declared in this arena. `Var(2)`/`Var(3)` are not coordinates
    /// — a lattice has two axes — and a value invariant across the lattice is
    /// exactly what an arbitrary subterm's stand-in should be.
    fn bind_metavars_past_the_axes(
        arena: &mut ExprBuilder,
        root: ExprRef,
        args: &[Uniform],
    ) -> ExprRef {
        let subs: Vec<(u8, ExprRef)> = args
            .iter()
            .enumerate()
            .map(|(i, u)| {
                let slot = arena.declare_uniform(u.decl());
                let axis = pixelflow_ir::decl::COORD_AXES as u8 + i as u8;
                (axis, arena.push_uniform(slot))
            })
            .collect();
        crate::egraph::template::substitute_vars(arena, root, &subs)
    }

    /// Evaluate `rule`'s LHS and RHS templates at every point in `points`
    /// and assert they agree within `tol` (relative for |value| > 1, else
    /// absolute) — the cross-form conditioned gate (§2.4). `tol` is per-rule:
    /// exact identities use a tight tolerance, estimate-based ones (`Recip`,
    /// `Rsqrt`) use a loose one, matching CLAUDE.md's own table.
    fn assert_oracle(rule: &dyn Rewrite, points: &[[f32; 4]], tol: f32) {
        // A metavariable stands for an arbitrary subterm. The first two can
        // be coordinates — a lattice has two axes — and the rest are the
        // kernel arguments they would be, sampled from the point's remaining
        // slots through the block. The same handles go into both arenas, so
        // the two sides read one value per metavariable.
        let args = [Uniform::new(0.0), Uniform::new(0.0)];
        let mut lhs_arena = ExprBuilder::new();
        let lhs_root = rule
            .lhs_template(&mut lhs_arena)
            .unwrap_or_else(|| panic!("{}: missing lhs_template", rule.name()));
        let lhs_root = bind_metavars_past_the_axes(&mut lhs_arena, lhs_root, &args);
        let mut rhs_arena = ExprBuilder::new();
        let rhs_root = rule
            .rhs_template(&mut rhs_arena)
            .unwrap_or_else(|| panic!("{}: missing rhs_template", rule.name()));
        let rhs_root = bind_metavars_past_the_axes(&mut rhs_arena, rhs_root, &args);

        for point in points {
            let values = [
                (args[0].identity(), point[2]),
                (args[1].identity(), point[3]),
            ];
            let coords = [point[0], point[1]];
            let lhs_bindings = BindingTable::empty()
                .bind_uniforms(lhs_arena.env(), &values)
                .expect("declared just above");
            let rhs_bindings = BindingTable::empty()
                .bind_uniforms(rhs_arena.env(), &values)
                .expect("declared just above");
            let lhs_term = Term::new(lhs_arena.node(lhs_root), lhs_arena.env());
            let rhs_term = Term::new(rhs_arena.node(rhs_root), rhs_arena.env());
            let lhs = eval_scalar(lhs_term, &coords, &lhs_bindings);
            let rhs = eval_scalar(rhs_term, &coords, &rhs_bindings);
            if lhs.is_nan() && rhs.is_nan() {
                continue;
            }
            let threshold = if lhs.abs() > 1.0 {
                tol * lhs.abs()
            } else {
                tol
            };
            assert!(
                (lhs - rhs).abs() <= threshold,
                "{}: LHS/RHS disagree at well-conditioned point {point:?}: \
                 lhs={lhs} rhs={rhs} (threshold {threshold})\n  lhs = {}\n  rhs = {}",
                rule.name(),
                pixelflow_ir::display(lhs_arena.node(lhs_root)),
                pixelflow_ir::display(rhs_arena.node(rhs_root)),
            );
        }
    }

    const TIGHT: f32 = 1e-4;
    /// `Recip`/`Rsqrt` are hardware estimates in the JIT, but `eval_scalar`
    /// computes them exactly (`1.0/x`, `1.0/sqrt(x)`) — see kind.rs
    /// `eval_unary`. The oracle here therefore checks the *exact* identity;
    /// the loose tolerance exists only for float rounding across the two
    /// independently-built expression trees, not for estimate error.
    const LOOSE: f32 = 1e-3;

    #[test]
    fn n1_n2_min_max_duality() {
        for r in min_max_duality_rules() {
            assert_oracle(r.as_ref(), &general_points(), TIGHT);
        }
    }

    #[test]
    fn n3_n4_min_max_absorption() {
        for r in min_max_absorption_rules() {
            assert_oracle(r.as_ref(), &general_points(), TIGHT);
        }
    }

    #[test]
    fn n5_n6_min_max_translate() {
        for r in min_max_translate_rules() {
            assert_oracle(r.as_ref(), &general_points(), TIGHT);
        }
    }

    #[test]
    fn n7_min_scaled_by_nonneg_literal() {
        assert_oracle(&MinScaledByNonnegLiteral, &general_points(), TIGHT);
    }

    #[test]
    fn n8_min_max_distributive() {
        assert_oracle(&MinMaxDistributive, &general_points(), TIGHT);
    }

    #[test]
    fn n9_n10_abs_as_lattice_op() {
        assert_oracle(&AbsAsMax, &general_points(), TIGHT);
        assert_oracle(&MaxSelfNegAsAbs, &general_points(), TIGHT);
    }

    #[test]
    fn n11_select_same_branch() {
        assert_oracle(&SelectSameBranch, &general_points(), TIGHT);
    }

    #[test]
    fn n12_n13_select_as_min_max() {
        assert_oracle(&SelectLtToMin, &general_points(), TIGHT);
        assert_oracle(&SelectLtToMax, &general_points(), TIGHT);
    }

    #[test]
    fn n14_select_hoist_unary() {
        for r in select_hoist_unary_rules() {
            // Var(0) is the mask metavariable here — must be a genuine
            // all-ones/all-zero pattern (select_mask_points), and Sqrt is in
            // the pool so the non-mask slots must stay positive.
            assert_oracle(r.as_ref(), &select_mask_points(), TIGHT);
        }
    }

    #[test]
    fn n15_compare_flip_lt() {
        assert_oracle(&CompareFlipLt, &general_points(), TIGHT);
    }

    #[test]
    fn n16_n17_tan_definition() {
        assert_oracle(&TanDefinition, &general_points(), TIGHT);
        assert_oracle(&TanFusion, &general_points(), TIGHT);
    }

    #[test]
    fn n18_exp_as_exp2() {
        assert_oracle(&ExpAsExp2, &general_points(), TIGHT);
    }

    #[test]
    fn n19_ln_as_log2() {
        assert_oracle(&LnAsLog2, &positive_points(), TIGHT);
    }

    #[test]
    fn n20_log10_as_log2() {
        assert_oracle(&Log10AsLog2, &positive_points(), TIGHT);
    }

    #[test]
    fn n21_sqrt_product() {
        assert_oracle(&SqrtProduct, &positive_points(), TIGHT);
    }

    #[test]
    fn n22_rsqrt_square_as_recip() {
        assert_oracle(&RsqrtSquareAsRecip, &positive_points(), LOOSE);
    }

    #[test]
    fn n23_normalize_as_sqrt() {
        assert_oracle(&NormalizeAsSqrt, &positive_points(), LOOSE);
    }

    #[test]
    fn n24_n24r_recip_product() {
        assert_oracle(&RecipProduct, &general_points(), LOOSE);
        assert_oracle(&RecipOfProduct, &general_points(), LOOSE);
    }

    #[test]
    fn n25_fma_unfuse() {
        assert_oracle(&FmaUnfuse, &general_points(), TIGHT);
    }

    #[test]
    fn n26_fma_identities() {
        assert_oracle(&FmaMulIdentity, &general_points(), TIGHT);
        assert_oracle(&FmaAddIdentity, &general_points(), TIGHT);
    }

    #[test]
    fn n27_neg_distributes() {
        assert_oracle(&NegDistributesAdd, &general_points(), TIGHT);
        assert_oracle(&NegDistributesMul, &general_points(), TIGHT);
    }

    #[test]
    fn n28_div_by_literal() {
        assert_oracle(&DivByLiteral, &general_points(), TIGHT);
    }

    /// Sanity: the batch is harness-only, and its count matches the doc
    /// comment on [`experimental_rules`] (guards against silent drift if a
    /// rule is added/removed without updating the comment).
    #[test]
    fn experimental_rules_count_and_names_are_unique() {
        let rules = experimental_rules();
        assert_eq!(rules.len(), 33, "unexpected experimental_rules() count");
        let mut names: Vec<&str> = rules.iter().map(|r| r.name()).collect();
        names.sort_unstable();
        let mut dedup = names.clone();
        dedup.dedup();
        assert_eq!(names, dedup, "duplicate rule name in experimental_rules()");
    }

    /// Saturation smoke test: `all_rules() + experimental_rules()` (93 rules)
    /// must saturate and extract without panicking on a handful of
    /// representative expressions exercising every family here (min/max,
    /// select, trig, exp/log, sqrt/rsqrt/recip, fma, neg, div-by-literal).
    /// This is a structural check on the new `RewriteAction` executors
    /// (graph.rs), not a semantics check — the per-family oracle tests above
    /// own semantics.
    #[test]
    fn experimental_rules_saturate_without_panicking() {
        use crate::egraph::{
            CostModel, EGraph, ENode, config_for_node_count, extract, saturate_with_full_budget,
        };
        use std::time::Duration;

        let costs = CostModel::latency_prior();

        // min(x, max(x, y)) — absorption + duality + select rules all apply.
        let build_min_max = |eg: &mut EGraph| {
            let x = eg.add(ENode::Var(0));
            let y = eg.add(ENode::Var(1));
            let max_xy = eg.add(ENode::Op {
                op: &ops::Max,
                children: vec![x, y],
            });
            eg.add(ENode::Op {
                op: &ops::Min,
                children: vec![x, max_xy],
            })
        };

        // tan(x) * cos(x) — tan/exp/log rules all apply somewhere in here.
        let build_tan_cos = |eg: &mut EGraph| {
            let x = eg.add(ENode::Var(0));
            let tan_x = eg.add(ENode::Op {
                op: &ops::Tan,
                children: vec![x],
            });
            let cos_x = eg.add(ENode::Op {
                op: &ops::Cos,
                children: vec![x],
            });
            eg.add(ENode::Op {
                op: &ops::Mul,
                children: vec![tan_x, cos_x],
            })
        };

        // fma(x, y, x/2) — fma unfuse/identities + div-by-literal.
        let build_fma_div = |eg: &mut EGraph| {
            let x = eg.add(ENode::Var(0));
            let y = eg.add(ENode::Var(1));
            let two = eg.add(ENode::constant(2.0));
            let x_half = eg.add(ENode::Op {
                op: &ops::Div,
                children: vec![x, two],
            });
            eg.add(ENode::Op {
                op: &ops::MulAdd,
                children: vec![x, y, x_half],
            })
        };

        // EGraph::with_rules takes ownership, so build one egraph per case
        // using a freshly constructed rule vector each time (Rewrite is not
        // Clone).
        type Builder = fn(&mut EGraph) -> EClassId;
        let cases: [(&str, Builder); 3] = [
            ("min-max", build_min_max),
            ("tan-cos", build_tan_cos),
            ("fma-div", build_fma_div),
        ];
        for (case_name, build) in cases {
            let mut rules = crate::math::all_rules();
            rules.extend(experimental_rules());
            let mut eg = EGraph::with_rules(rules);
            let root = build(&mut eg);
            let class_cap = config_for_node_count(8).max_classes;
            let _ = saturate_with_full_budget(&mut eg, 200, class_cap, Duration::from_secs(10));
            // The point of this test is that saturate+extract complete
            // without panicking under the full 62+experimental rule set; a
            // cost of exactly 0 is a legitimate outcome (e.g. min-max
            // absorption reducing the whole expression to a bare Var, whose
            // latency_prior cost is 0), not a failure.
            let (arena, _env, _cost) = extract(&eg, eg.find(root), &costs);
            assert!(
                !format!("{}", pixelflow_ir::display(arena.entry())).is_empty(),
                "{case_name}: extraction produced an empty expression"
            );
        }
    }

    /// `experimental_rules()` must never be reachable from `all_rules()`
    /// (production behavior is untouched by this batch's existence).
    #[test]
    fn experimental_rules_not_in_all_rules() {
        let experimental_rules_vec = experimental_rules();
        let experimental_names: std::collections::HashSet<&str> =
            experimental_rules_vec.iter().map(|r| r.name()).collect();
        for r in crate::math::all_rules() {
            assert!(
                !experimental_names.contains(r.name()),
                "all_rules() must not contain a Round-2 experimental rule name: {}",
                r.name()
            );
        }
    }
}
