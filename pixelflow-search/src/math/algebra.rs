//! Core algebraic rewrite rules.
//!
//! This module contains:
//! - `InversePair` trait: derives 4 rules from one trait impl (canonicalize, involution, cancellation, annihilation)
//! - Basic algebraic rules: commutativity, identity, annihilator, associativity, idempotent
//! - Distributivity and factoring

use std::marker::PhantomData;

use crate::arena_pat;
use crate::egraph::{EClassId, EGraph, ENode, Rewrite, RewriteAction, RootShapes};
use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_ir::{Lit, OpKind};

// ============================================================================
// InversePair: The Core Algebraic Relationship
// ============================================================================

/// A complete inverse relationship between operations.
///
/// An inverse pair captures the full algebraic structure:
/// - BASE: The fundamental binary operation (Add, Mul)
/// - INVERSE: The unary inverse operation (Neg, Recip)
/// - DERIVED: Syntactic sugar for BASE(a, INVERSE(b)) (Sub, Div)
/// - IDENTITY: The identity element for BASE (0, 1)
///
/// From one InversePair, we derive four rewrite rules:
/// - Canonicalize: a ⊖ b → a ⊕ inv(b)
/// - Involution: inv(inv(x)) → x
/// - Cancellation: (x ⊕ a) ⊖ a → x
/// - InverseAnnihilation: x ⊕ inv(x) → identity
pub trait InversePair: Send + Sync {
    /// The base operation (Add, Mul)
    fn base() -> OpKind;
    /// The inverse operation (Neg, Recip)
    fn inverse() -> OpKind;
    /// The derived operation (Sub, Div)
    fn derived() -> OpKind;
    /// The identity element (0.0 for Add, 1.0 for Mul)
    fn identity() -> f32;
}

/// Addition and Negation are inverses.
/// - x + neg(x) = 0
/// - neg(neg(x)) = x
/// - a - b = a + neg(b)
/// - (x + a) - a = x
pub struct AddNeg;
impl InversePair for AddNeg {
    fn base() -> OpKind {
        OpKind::Add
    }
    fn inverse() -> OpKind {
        OpKind::Neg
    }
    fn derived() -> OpKind {
        OpKind::Sub
    }
    fn identity() -> f32 {
        0.0
    }
}

/// Multiplication and Reciprocal are inverses.
/// - x * recip(x) = 1
/// - recip(recip(x)) = x
/// - a / b = a * recip(b)
/// - (x * a) / a = x
pub struct MulRecip;
impl InversePair for MulRecip {
    fn base() -> OpKind {
        OpKind::Mul
    }
    fn inverse() -> OpKind {
        OpKind::Recip
    }
    fn derived() -> OpKind {
        OpKind::Div
    }
    fn identity() -> f32 {
        1.0
    }
}

// ============================================================================
// Rules Derived from InversePair
// ============================================================================

/// Canonicalize: a ⊖ b → a ⊕ inv(b)
///
/// Reduces the operator set by expressing derived ops in terms of base + inverse.
/// - `Canonicalize::<AddNeg>`: a - b → a + neg(b)
/// - `Canonicalize::<MulRecip>`: a / b → a * recip(b)
pub struct Canonicalize<T: InversePair>(PhantomData<T>);

impl<T: InversePair> Canonicalize<T> {
    pub fn new() -> Box<Self> {
        Box::new(Self(PhantomData))
    }
}

impl<T: InversePair> Default for Canonicalize<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T: InversePair> Rewrite for Canonicalize<T> {
    fn name(&self) -> &str {
        "canonicalize"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(T::derived())
    }

    fn specialization(&self) -> Option<pixelflow_ir::OpKind> {
        Some(T::derived())
    }

    fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if node.op() != Some(T::derived()) {
            return None;
        }
        let (a, b) = node.binary_operands()?;

        Some(RewriteAction::Canonicalize {
            target: T::base(),
            inverse: T::inverse(),
            a,
            b,
        })
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin T::derived(), (var 0), (var 1)))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin T::base(), (var 0), (un T::inverse(), (var 1))))
    }
}

/// Involution: inv(inv(x)) → x
///
/// The unary inverse is its own inverse.
/// - `Involution::<AddNeg>`: neg(neg(x)) → x
/// - `Involution::<MulRecip>`: recip(recip(x)) → x
pub struct Involution<T: InversePair>(PhantomData<T>);

impl<T: InversePair> Involution<T> {
    pub fn new() -> Box<Self> {
        Box::new(Self(PhantomData))
    }
}

impl<T: InversePair> Default for Involution<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T: InversePair> Rewrite for Involution<T> {
    fn name(&self) -> &str {
        "involution"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(T::inverse())
    }

    fn specialization(&self) -> Option<pixelflow_ir::OpKind> {
        Some(T::derived())
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if node.op() != Some(T::inverse()) {
            return None;
        }

        let children = node.children();
        if children.len() != 1 {
            return None;
        }
        let inner_id = children[0];

        for inner_node in egraph.nodes(inner_id) {
            if inner_node.op() == Some(T::inverse()) {
                let inner_children = inner_node.children();
                if inner_children.len() == 1 {
                    return Some(RewriteAction::Union(inner_children[0]));
                }
            }
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        // Inverse(Inverse(V0))
        let inv = T::inverse();
        Some(arena_pat!(__a, un inv, (un inv, (var 0))))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, var 0))
    }
}

/// Cancellation: (x ⊕ a) ⊖ a → x
///
/// The derived op cancels the base op when applied to the same operand.
/// - `Cancellation::<AddNeg>`: (x + a) - a → x
/// - `Cancellation::<MulRecip>`: (x * a) / a → x
pub struct Cancellation<T: InversePair>(PhantomData<T>);

impl<T: InversePair> Cancellation<T> {
    pub fn new() -> Box<Self> {
        Box::new(Self(PhantomData))
    }
}

impl<T: InversePair> Default for Cancellation<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T: InversePair> Rewrite for Cancellation<T> {
    fn name(&self) -> &str {
        "cancellation"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(T::derived())
    }

    fn specialization(&self) -> Option<pixelflow_ir::OpKind> {
        Some(T::derived())
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if node.op() != Some(T::derived()) {
            return None;
        }
        let (numerator, canceller) = node.binary_operands()?;

        for inner_node in egraph.nodes(numerator) {
            if inner_node.op() == Some(T::base())
                && let Some((a, b)) = inner_node.binary_operands()
            {
                // (a ⊕ b) ⊖ b → a
                if egraph.find(b) == egraph.find(canceller) {
                    return Some(RewriteAction::Union(a));
                }
                // (b ⊕ a) ⊖ b → a (if BASE is commutative)
                if T::base().is_commutative() && egraph.find(a) == egraph.find(canceller) {
                    return Some(RewriteAction::Union(b));
                }
            }
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin T::derived(), (bin T::base(), (var 0), (var 1)), (var 1)))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, var 0))
    }
}

/// InverseAnnihilation: x ⊕ inv(x) → identity
///
/// An element combined with its inverse yields the identity.
/// - `InverseAnnihilation::<AddNeg>`: x + neg(x) → 0
/// - `InverseAnnihilation::<MulRecip>`: x * recip(x) → 1
pub struct InverseAnnihilation<T: InversePair>(PhantomData<T>);

impl<T: InversePair> InverseAnnihilation<T> {
    pub fn new() -> Box<Self> {
        Box::new(Self(PhantomData))
    }
}

impl<T: InversePair> Default for InverseAnnihilation<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T: InversePair> Rewrite for InverseAnnihilation<T> {
    fn name(&self) -> &str {
        "inverse-annihilation"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(T::base())
    }

    fn specialization(&self) -> Option<pixelflow_ir::OpKind> {
        Some(T::derived())
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if node.op() != Some(T::base()) {
            return None;
        }
        let (a, b) = node.binary_operands()?;

        // x ⊕ inv(x) → identity
        for node_b in egraph.nodes(b) {
            if node_b.op() == Some(T::inverse())
                && let Some(&inner) = node_b.children().first()
                && egraph.find(inner) == egraph.find(a)
            {
                return Some(RewriteAction::Create(ENode::constant(T::identity())));
            }
        }

        // inv(x) ⊕ x → identity
        for node_a in egraph.nodes(a) {
            if node_a.op() == Some(T::inverse())
                && let Some(&inner) = node_a.children().first()
                && egraph.find(inner) == egraph.find(b)
            {
                return Some(RewriteAction::Create(ENode::constant(T::identity())));
            }
        }

        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin T::base(), (var 0), (un T::inverse(), (var 0))))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, cst T::identity()))
    }
}

// ============================================================================
// Basic Algebraic Rules (not derived from InversePair)
// ============================================================================

/// Associativity: (a op b) op c → a op (b op c)
pub struct Associative {
    op: OpKind,
}

impl Associative {
    pub fn new(op: OpKind) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Associative {
    fn name(&self) -> &str {
        "associative"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(self.op)
    }

    fn specialization(&self) -> Option<pixelflow_ir::OpKind> {
        Some(self.op)
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op != self.op {
            return None;
        }

        let (left, right) = node.binary_operands()?;

        for child in egraph.nodes(left) {
            let Some(child_op) = child.op() else { continue };
            if child_op != self.op {
                continue;
            }
            let Some((a, b)) = child.binary_operands() else {
                continue;
            };

            return Some(RewriteAction::Associate {
                op: self.op,
                a,
                b,
                c: right,
            });
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        // Op(Op(V0, V1), V2)
        let k = self.op;
        Some(arena_pat!(__a, bin k, (bin k, (var 0), (var 1)), (var 2)))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        // Op(V0, Op(V1, V2))
        let k = self.op;
        Some(arena_pat!(__a, bin k, (var 0), (bin k, (var 1), (var 2))))
    }
}

/// Reverse associativity: a op (b op c) → (a op b) op c
pub struct ReverseAssociative {
    op: OpKind,
}

impl ReverseAssociative {
    pub fn new(op: OpKind) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for ReverseAssociative {
    fn name(&self) -> &str {
        "reverse-associative"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(self.op)
    }

    fn specialization(&self) -> Option<pixelflow_ir::OpKind> {
        Some(self.op)
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op != self.op {
            return None;
        }

        let (left, right) = node.binary_operands()?;

        // Check if the right child has a node with the same op
        for child in egraph.nodes(right) {
            let Some(child_op) = child.op() else { continue };
            if child_op != self.op {
                continue;
            }
            let Some((b, c)) = child.binary_operands() else {
                continue;
            };

            return Some(RewriteAction::ReverseAssociate {
                op: self.op,
                a: left,
                b,
                c,
            });
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        // Op(V0, Op(V1, V2))
        let k = self.op;
        Some(arena_pat!(__a, bin k, (var 0), (bin k, (var 1), (var 2))))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        // Op(Op(V0, V1), V2)
        let k = self.op;
        Some(arena_pat!(__a, bin k, (bin k, (var 0), (var 1)), (var 2)))
    }
}

/// Commutativity: a op b → b op a
pub struct Commutative {
    op: OpKind,
}

impl Commutative {
    pub fn new(op: OpKind) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Commutative {
    fn name(&self) -> &str {
        "commutative"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(self.op)
    }

    fn specialization(&self) -> Option<pixelflow_ir::OpKind> {
        Some(self.op)
    }

    fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op != self.op {
            return None;
        }

        let (a, b) = node.binary_operands()?;
        if a == b {
            return None;
        }

        let swapped = ENode::apply(self.op, &[b, a]);
        Some(RewriteAction::Create(swapped))
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin self.op, (var 0), (var 1)))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin self.op, (var 1), (var 0)))
    }
}

/// Distributivity: A * (B + C) → A*B + A*C
pub struct Distributive {
    outer: OpKind,
    inner: OpKind,
}

impl Distributive {
    pub fn new(outer: OpKind, inner: OpKind) -> Box<Self> {
        Box::new(Self { outer, inner })
    }
}

impl Rewrite for Distributive {
    fn name(&self) -> &str {
        "distribute"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(self.outer)
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op != self.outer {
            return None;
        }

        let (a, other) = node.binary_operands()?;

        for child_node in egraph.nodes(other) {
            let Some(child_op) = child_node.op() else {
                continue;
            };
            if child_op != self.inner {
                continue;
            }
            let Some((b, c)) = child_node.binary_operands() else {
                continue;
            };

            return Some(RewriteAction::Distribute {
                outer: self.outer,
                inner: self.inner,
                a,
                b,
                c,
            });
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin self.outer, (var 0), (bin self.inner, (var 1), (var 2))))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        // Inner(Outer(V0, V1), Outer(V0, V2))
        let ok = self.outer;
        let ik = self.inner;
        Some(arena_pat!(__a, bin ik, (bin ok, (var 0), (var 1)), (bin ok, (var 0), (var 2))))
    }
}

/// Factoring: A*B + A*C → A * (B + C)
pub struct Factor {
    outer: OpKind,
    inner: OpKind,
}

impl Factor {
    pub fn new(outer: OpKind, inner: OpKind) -> Box<Self> {
        Box::new(Self { outer, inner })
    }
}

impl Rewrite for Factor {
    fn name(&self) -> &str {
        "factor"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(self.outer)
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op != self.outer {
            return None;
        }

        let (left, right) = node.binary_operands()?;

        for l_node in egraph.nodes(left) {
            let l_op = l_node.op()?;
            if l_op != self.inner {
                continue;
            }
            let (la, lb) = l_node.binary_operands()?;

            for r_node in egraph.nodes(right) {
                let r_op = r_node.op()?;
                if r_op != self.inner {
                    continue;
                }
                let (ra, rb) = r_node.binary_operands()?;

                let (common, unique_l, unique_r) = if egraph.find(la) == egraph.find(ra) {
                    (la, lb, rb)
                } else if egraph.find(la) == egraph.find(rb) {
                    (la, lb, ra)
                } else if egraph.find(lb) == egraph.find(ra) {
                    (lb, la, rb)
                } else if egraph.find(lb) == egraph.find(rb) {
                    (lb, la, ra)
                } else {
                    continue;
                };

                return Some(RewriteAction::Factor {
                    outer: self.outer,
                    inner: self.inner,
                    common,
                    unique_l,
                    unique_r,
                });
            }
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        // Outer(Inner(V0, V1), Inner(V0, V2))
        let ok = self.outer;
        let ik = self.inner;
        Some(arena_pat!(__a, bin ok, (bin ik, (var 0), (var 1)), (bin ik, (var 0), (var 2))))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin self.inner, (var 0), (bin self.outer, (var 1), (var 2))))
    }
}

/// Identity: x op identity → x
pub struct Identity {
    op: OpKind,
}

impl Identity {
    pub fn new(op: OpKind) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Identity {
    fn name(&self) -> &str {
        "identity"
    }

    fn root_shapes(&self) -> RootShapes {
        if self.op.identity().is_none() {
            return RootShapes::nothing();
        }
        RootShapes::kind(self.op)
    }

    fn specialization(&self) -> Option<pixelflow_ir::OpKind> {
        Some(self.op)
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op != self.op {
            return None;
        }

        let id_val = self.op.identity()?;
        let (a, b) = node.binary_operands()?;

        if egraph.contains_const(b, id_val) {
            return Some(RewriteAction::Union(a));
        }
        if egraph.contains_const(a, id_val) {
            return Some(RewriteAction::Union(b));
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        // Op(V0, Const(identity))
        let id_val = self.op.identity()?;
        Some(arena_pat!(__a, bin self.op, (var 0), (cst id_val)))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, var 0))
    }
}

/// Annihilator: x op annihilator → annihilator
pub struct Annihilator {
    op: OpKind,
}

impl Annihilator {
    pub fn new(op: OpKind) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Annihilator {
    fn name(&self) -> &str {
        "annihilator"
    }

    fn root_shapes(&self) -> RootShapes {
        if self.op.annihilator().is_none() {
            return RootShapes::nothing();
        }
        RootShapes::kind(self.op)
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op != self.op {
            return None;
        }

        let zero_val = self.op.annihilator()?;
        let (a, b) = node.binary_operands()?;

        if egraph.contains_const(a, zero_val) || egraph.contains_const(b, zero_val) {
            return Some(RewriteAction::Create(ENode::constant(zero_val)));
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        // Op(V0, Const(annihilator))
        let ann = self.op.annihilator()?;
        Some(arena_pat!(__a, bin self.op, (var 0), (cst ann)))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        // Const(annihilator)
        let ann = self.op.annihilator()?;
        Some(arena_pat!(__a, cst ann))
    }
}

/// Idempotence: x op x → x
pub struct Idempotent {
    op: OpKind,
}

impl Idempotent {
    pub fn new(op: OpKind) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Idempotent {
    fn name(&self) -> &str {
        "idempotent"
    }

    fn root_shapes(&self) -> RootShapes {
        if !self.op.is_idempotent() {
            return RootShapes::nothing();
        }
        RootShapes::kind(self.op)
    }

    fn specialization(&self) -> Option<pixelflow_ir::OpKind> {
        Some(self.op)
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op != self.op {
            return None;
        }
        if !self.op.is_idempotent() {
            return None;
        }

        let (a, b) = node.binary_operands()?;

        if egraph.find(a) == egraph.find(b) {
            return Some(RewriteAction::Union(a));
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin self.op, (var 0), (var 0)))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, var 0))
    }
}

// ============================================================================
// Doubling: a + a ↔ 2 * a
// ============================================================================

/// Doubling: a + a → 2 * a
///
/// This is a special case of the general pattern N * a = Sum(a, N).
/// For now we only handle the N=2 case which is common in trig identities.
///
/// Combined with the inverse (halving), this enables:
///   - sin(x)*cos(x) + cos(x)*sin(x) = 2*sin(x)*cos(x) = sin(2x)
///   - Therefore sin(x)*cos(x) = sin(2x)/2
pub struct Doubling;

impl Doubling {
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Rewrite for Doubling {
    fn name(&self) -> &str {
        "doubling"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(OpKind::Add)
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: Add(a, b) where a == b
        let &ENode::Op { op, children } = node else {
            return None;
        };
        if op != OpKind::Add {
            return None;
        }
        if children.len() != 2 {
            return None;
        }

        let a = children[0];
        let b = children[1];

        // Check if a == b (same e-class)
        if egraph.find(a) == egraph.find(b) {
            // a + a → 2 * a
            return Some(RewriteAction::Doubling { a });
        }

        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin OpKind::Add, (var 0), (var 0)))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin OpKind::Mul, (cst 2.0), (var 0)))
    }
}

/// Halving: 2 * a → a + a (reverse of doubling)
///
/// This allows the e-graph to explore both representations.
pub struct Halving;

impl Halving {
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Rewrite for Halving {
    fn name(&self) -> &str {
        "halving"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(OpKind::Mul)
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: Mul(2, a) or Mul(a, 2)
        let &ENode::Op { op, children } = node else {
            return None;
        };
        if op != OpKind::Mul {
            return None;
        }
        if children.len() != 2 {
            return None;
        }

        let left = children[0];
        let right = children[1];

        // Check for 2 * a
        if egraph.contains_const(left, 2.0) {
            return Some(RewriteAction::Halving { a: right });
        }
        // Check for a * 2
        if egraph.contains_const(right, 2.0) {
            return Some(RewriteAction::Halving { a: left });
        }

        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin OpKind::Mul, (cst 2.0), (var 0)))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin OpKind::Add, (var 0), (var 0)))
    }
}

// ============================================================================
// Constant Folding
// ============================================================================

/// The most arguments any operation takes: the widest [`OpKind::arity`]
/// (`MulAdd`, `If`, `Gather`), and the length of the buffer [`ConstantFold`]
/// gathers them into. `every_operation_fits_the_folders_argument_buffer`
/// holds it to that.
const MAX_FOLD_ARGUMENTS: usize = 3;

/// Constant folding: op(const, ...) → const
///
/// Evaluates operations on constant arguments at compile time.
/// This is essential for rules like Canonicalize to work fully:
/// - `X / 2` → `X * recip(2)` → `X * 0.5` (requires folding recip(2))
pub struct ConstantFold;

impl ConstantFold {
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Rewrite for ConstantFold {
    fn name(&self) -> &str {
        "constant-fold"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::every_op()
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // The operation and its arguments, each a constant. A shift's count
        // is read off the node, where it is already a valid count, and is
        // an argument all the same: the folder asks the one question of
        // `[x, count]` that it asks of every operation. The arguments are a
        // prefix of an inline array, as a node's operands are: no operation
        // takes more than `MAX_FOLD_ARGUMENTS`, so the prefix always fits.
        let mut buffer = [Lit::Real(0.0); MAX_FOLD_ARGUMENTS];
        let (op, arity) = match node {
            ENode::Op { op, children } => {
                for (slot, &child_id) in buffer.iter_mut().zip(children.iter()) {
                    *slot = constant_of(egraph, child_id)?;
                }
                (*op, children.len())
            }
            ENode::Shift {
                direction,
                count,
                operand,
            } => {
                buffer[0] = constant_of(egraph, *operand)?;
                buffer[1] = Lit::Real(count.literal());
                let op = direction.op();
                (op, op.arity())
            }
            _ => return None,
        };
        let args = &buffer[..arity];

        // Refuse to fold anything whose answer is platform-specific. This
        // matters more than a same-machine consistency nicety: the folder runs
        // on the BUILD host at macro-expansion time, while the folded constant
        // executes on the TARGET, so folding here bakes the build machine's
        // arithmetic into code that may run somewhere that computes
        // differently. `fold_is_platform_specific` owns the classification (NaN
        // and signed-zero Min/Max, NaN Gt/Ge, Round at a tie, the reciprocal
        // estimates, invalid TruncToInt) so it lives in one place next to the
        // semantics rather than as conditions here. `MulAdd` is deliberately
        // not on the list: one rounding versus two is precision, which the
        // language puts on the table, not a different value.
        //
        // Complete for `Round`/`Recip`/`Rsqrt`/`TruncToInt` (unary — no rewrite
        // can reintroduce the fold). Partial for `Min`/`Max`: the e-graph
        // installs commutativity for them, so a commuted form can still be
        // folded. That is permitted, since the result is unspecified, but this
        // removes the direct path.
        //
        // The rows are about real values; a mask operand has one answer on
        // every target, so an argument list with one in it is never refused
        // here (and `eval_*` decides what the op makes of it).
        let mut reals = [0.0f32; MAX_FOLD_ARGUMENTS];
        if let Some(reals) = reals_of(args, &mut reals)
            && op.fold_is_platform_specific(reals)
        {
            return None;
        }

        // Evaluate based on arity
        let result = match *args {
            [x] => op.eval_unary(x)?,
            [x, y] => op.eval_binary(x, y)?,
            [x, y, z] => op.eval_ternary(x, y, z)?,
            _ => return None,
        };

        // Don't bake in a non-finite result of ARITHMETIC: `1e38 * 1e38` folding
        // to `inf` freezes an overflow no source constant asked for. A mask is
        // not a float and the guard never sees one; what is still exempt is the
        // bitwise domain whose results ride in a real lane by convention — the
        // integer-domain primitives exp/log lower to, building exponents out of
        // raw bit patterns, and `If`, which hands an arm through. Without the
        // exemption this guard silently switches off exp/log folding while
        // looking like a safety check.
        if let Lit::Real(real) = result
            && !real.is_finite()
            && !op.is_bitwise_domain()
        {
            return None;
        }

        Some(RewriteAction::Create(ENode::Const(result)))
    }
}

/// The first literal in `class`, if it holds one.
fn constant_of(egraph: &EGraph, class: EClassId) -> Option<Lit> {
    egraph.nodes(class).find_map(ENode::literal)
}

/// `args` as reals, into `out`, when every one of them is a real.
fn reals_of<'a>(args: &[Lit], out: &'a mut [f32; MAX_FOLD_ARGUMENTS]) -> Option<&'a [f32]> {
    for (slot, lit) in out.iter_mut().zip(args) {
        *slot = lit.real()?;
    }
    Some(&out[..args.len()])
}

// ============================================================================
// Rule Collection
// ============================================================================

/// All algebraic rules derived from InversePair trait.
pub fn inverse_pair_rules() -> Vec<Box<dyn Rewrite>> {
    vec![
        // AddNeg: a - b → a + neg(b), neg(neg(x)) → x, etc.
        Canonicalize::<AddNeg>::new(),
        Involution::<AddNeg>::new(),
        Cancellation::<AddNeg>::new(),
        InverseAnnihilation::<AddNeg>::new(),
        // MulRecip: a / b → a * recip(b), recip(recip(x)) → x, etc.
        Canonicalize::<MulRecip>::new(),
        Involution::<MulRecip>::new(),
        Cancellation::<MulRecip>::new(),
        InverseAnnihilation::<MulRecip>::new(),
    ]
}

/// Basic algebraic rules (commutativity, identity, annihilator, etc.).
pub fn basic_algebra_rules() -> Vec<Box<dyn Rewrite>> {
    vec![
        // Constant folding: op(const, ...) → const
        // MUST come first to enable other rules to work fully
        ConstantFold::new(),
        // Commutativity
        Commutative::new(OpKind::Add),
        Commutative::new(OpKind::Mul),
        Commutative::new(OpKind::Min),
        Commutative::new(OpKind::Max),
        // Identity elements: x + 0 → x, x * 1 → x
        Identity::new(OpKind::Add),
        Identity::new(OpKind::Mul),
        // Annihilators: x * 0 → 0
        Annihilator::new(OpKind::Mul),
        // Idempotent: min(x,x) → x, max(x,x) → x
        Idempotent::new(OpKind::Min),
        Idempotent::new(OpKind::Max),
        // Distributivity: a * (b + c) → a*b + a*c
        Distributive::new(OpKind::Mul, OpKind::Add),
        // Factoring: a*b + a*c → a * (b + c)
        Factor::new(OpKind::Add, OpKind::Mul),
        // Doubling/Halving: a + a ↔ 2 * a
        Doubling::new(),
        Halving::new(),
        // Associativity (L→R): (a op b) op c → a op (b op c)
        Associative::new(OpKind::Add),
        Associative::new(OpKind::Mul),
        Associative::new(OpKind::Min),
        Associative::new(OpKind::Max),
        // Reverse associativity (R→L): a op (b op c) → (a op b) op c
        ReverseAssociative::new(OpKind::Add),
        ReverseAssociative::new(OpKind::Mul),
        ReverseAssociative::new(OpKind::Min),
        ReverseAssociative::new(OpKind::Max),
    ]
}

/// All algebra rules combined.
pub fn algebra_rules() -> Vec<Box<dyn Rewrite>> {
    let mut rules = inverse_pair_rules();
    rules.extend(basic_algebra_rules());
    rules
}

#[cfg(test)]
mod platform_specific_fold_tests {
    use super::*;
    use crate::egraph::rewrite::{Rewrite, RewriteAction};
    use crate::egraph::{EGraph, ENode};

    /// Drive `ConstantFold::apply` directly on `op(lits...)` and report the
    /// literal it folded to, if any.
    fn folds_lit(op: OpKind, args: &[Lit]) -> Option<Lit> {
        let mut eg = EGraph::new();
        let children: Vec<_> = args.iter().map(|&v| eg.add(ENode::Const(v))).collect();
        let node = ENode::apply(op, &children);
        let id = eg.add(node);
        match ConstantFold.apply(&eg, id, &node) {
            Some(RewriteAction::Create(ENode::Const(lit))) => Some(lit),
            _ => None,
        }
    }

    /// `folds_lit` over reals, for the arithmetic rows.
    fn folds(op: OpKind, args: &[f32]) -> Option<f32> {
        let args: Vec<Lit> = args.iter().map(|&v| Lit::Real(v)).collect();
        folds_lit(op, &args).and_then(Lit::real)
    }

    /// The folder gathers an operation's arguments into a buffer of
    /// `MAX_FOLD_ARGUMENTS`; a wider operation would have nowhere to go, so
    /// the day one exists this fails instead of the folder.
    #[test]
    fn every_operation_fits_the_folders_argument_buffer() {
        for kind in OpKind::all() {
            assert!(
                kind.arity() <= MAX_FOLD_ARGUMENTS,
                "{kind:?} takes {} arguments, past the folder's {MAX_FOLD_ARGUMENTS}",
                kind.arity()
            );
        }
    }

    /// A shift folds over a constant operand by the count it carries, to the
    /// value the op evaluates to, in either direction; an operand that is not
    /// a constant leaves nothing to fold.
    #[test]
    fn a_shift_folds_by_the_count_its_node_carries() {
        use crate::egraph::ShiftDirection;
        use pixelflow_ir::ShiftCount;

        let operand = 1.5f32;
        for direction in [ShiftDirection::Left, ShiftDirection::Right] {
            for bits in 0..ShiftCount::WORD_BITS {
                let count = ShiftCount::from_literal(bits as f32).expect("in range");
                let mut eg = EGraph::new();
                let constant = eg.add(ENode::constant(operand));
                let node = ENode::Shift {
                    direction,
                    count,
                    operand: constant,
                };
                let id = eg.add(node);
                let folded = match ConstantFold.apply(&eg, id, &node) {
                    Some(RewriteAction::Create(ENode::Const(lit))) => Some(lit),
                    _ => None,
                };
                let expected = direction
                    .op()
                    .eval_binary(Lit::Real(operand), Lit::Real(count.literal()));
                assert_eq!(folded, expected, "{direction:?} by {bits}");
            }
        }

        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0));
        let node = ENode::Shift {
            direction: ShiftDirection::Left,
            count: ShiftCount::from_literal(3.0).expect("in range"),
            operand: x,
        };
        let id = eg.add(node);
        assert!(ConstantFold.apply(&eg, id, &node).is_none());
    }

    /// The folder runs on the BUILD host; the constant it bakes in executes on
    /// the TARGET. For ops whose answer differs between targets it must decline,
    /// or a cross-compile silently ships the build machine's arithmetic.
    #[test]
    fn declines_platform_specific_rounding_ties() {
        // x86 nearest-even says 2, aarch64 FRINTA says 3 — so neither.
        assert_eq!(folds(OpKind::Round, &[2.5]), None);
        assert_eq!(folds(OpKind::Round, &[-1.5]), None);
        assert_eq!(folds(OpKind::Round, &[0.5]), None);
        // Away from a tie every target agrees, so folding is still allowed.
        assert_eq!(folds(OpKind::Round, &[2.4]), Some(2.0));
        assert_eq!(folds(OpKind::Round, &[2.6]), Some(3.0));
    }

    /// Non-tie inputs in `[-0.5, -0.0]` round to `-0.0` in both JIT tiers but
    /// to `+0.0` through the combinator tier's `(x + 0.5).floor()`, so the
    /// folder must decline there too — `1.0 / x` makes the difference `-inf`
    /// versus `+inf`.
    #[test]
    fn declines_round_folds_that_lose_the_sign_of_zero() {
        assert_eq!(folds(OpKind::Round, &[-0.2]), None);
        assert_eq!(folds(OpKind::Round, &[-0.4]), None);
        assert_eq!(folds(OpKind::Round, &[-0.0]), None);
        // Past -0.5 the result is -1.0 in every tier, so folding is fine again.
        assert_eq!(folds(OpKind::Round, &[-0.6]), Some(-1.0));
        // And positive inputs never had the ambiguity.
        assert_eq!(
            folds(OpKind::Round, &[0.2]).map(f32::to_bits),
            Some(0.0f32.to_bits())
        );
    }

    #[test]
    fn declines_platform_specific_nan_min_max() {
        let nan = f32::NAN;
        // x86 yields the second operand, aarch64 propagates the NaN.
        assert_eq!(folds(OpKind::Min, &[nan, 1.0]), None);
        assert_eq!(folds(OpKind::Min, &[1.0, nan]), None);
        assert_eq!(folds(OpKind::Max, &[nan, 1.0]), None);
        // Without a NaN the targets agree and folding proceeds.
        assert_eq!(folds(OpKind::Min, &[1.0, 2.0]), Some(1.0));
        assert_eq!(folds(OpKind::Max, &[1.0, 2.0]), Some(2.0));
    }

    /// Opposite-signed zeros are the non-NaN case where Min/Max still diverge:
    /// x86 picks by operand order (`-0.0 < 0.0` is false, so the second wins),
    /// aarch64 `FMIN` picks `-0.0` and `FMAX` picks `+0.0`. `==` cannot tell
    /// them apart, but `1.0 / x` makes it `+inf` versus `-inf`.
    #[test]
    fn declines_signed_zero_min_max() {
        assert_eq!(folds(OpKind::Min, &[-0.0, 0.0]), None);
        assert_eq!(folds(OpKind::Min, &[0.0, -0.0]), None);
        assert_eq!(folds(OpKind::Max, &[-0.0, 0.0]), None);
        // Same-signed zeros are indistinguishable, so folding is fine.
        assert_eq!(
            folds(OpKind::Min, &[0.0, 0.0]).map(f32::to_bits),
            Some(0.0f32.to_bits())
        );
    }

    /// `Recip`/`Rsqrt` are estimate instructions, and each target estimates
    /// differently: `rcpps` gives ~12 bits, `vrcp14ps` ~14, aarch64 runs
    /// `FRECPE` plus a `FRECPS` Newton step. `recip(3.0)` executes as
    /// `0.33325195` on SSE2 against the exact `0.33333334` the folder would
    /// compute — so unlike the cases above there is no argument that is safe.
    #[test]
    fn never_folds_reciprocal_estimates() {
        for x in [3.0f32, 2.0, 1.0, 0.5, 16.0] {
            assert_eq!(
                folds(OpKind::Recip, &[x]),
                None,
                "recip({x}) is an estimate"
            );
            assert_eq!(
                folds(OpKind::Rsqrt, &[x]),
                None,
                "rsqrt({x}) is an estimate"
            );
        }
    }

    /// `MulAdd` always folds, and the fold rounds once. One rounding (an FMA
    /// instruction) versus two (`mulps` then `addps`, or a decomposition under
    /// register pressure) is precision inside the language's contract, never a
    /// different value, so no argument is refused.
    #[test]
    fn folds_muladd_unconditionally_with_one_rounding() {
        // One rounding gives 8194.001, two give 8194.0: the fold is the former.
        assert_eq!(
            folds(OpKind::MulAdd, &[1.000_000_1, 4097.0, 4097.0]).map(f32::to_bits),
            Some(libm::fmaf(1.000_000_1, 4097.0, 4097.0).to_bits())
        );
        assert_eq!(folds(OpKind::MulAdd, &[2.0, 3.0, 4.0]), Some(10.0));
    }

    /// A folded comparison is a mask, the literal every backend tabulates as
    /// the all-ones or all-zero lane an executed comparison writes — never a
    /// float, so a folded `1.0` can no longer blend `7.0` against `9.0` into
    /// `4.5`.
    #[test]
    fn comparison_folds_produce_masks() {
        let (t, f) = (Lit::Mask(true), Lit::Mask(false));
        let real = |v: f32| Lit::Real(v);
        assert_eq!(folds_lit(OpKind::Lt, &[real(1.0), real(2.0)]), Some(t));
        assert_eq!(folds_lit(OpKind::Lt, &[real(2.0), real(1.0)]), Some(f));
        assert_eq!(folds_lit(OpKind::Eq, &[real(2.0), real(2.0)]), Some(t));
        assert_eq!(folds_lit(OpKind::Ne, &[real(2.0), real(2.0)]), Some(f));
        // And such a mask selects an arm whole. (`If` is the consumer this
        // test folds; the end-to-end path through a real `and` is covered by
        // `pixelflow-codegen/tests/transcendental_jit.rs`.)
        assert_eq!(
            folds_lit(OpKind::If, &[t, real(7.0), real(9.0)]),
            Some(real(7.0))
        );
        assert_eq!(
            folds_lit(OpKind::If, &[f, real(7.0), real(9.0)]),
            Some(real(9.0))
        );
        // The two quantifiers' identities are masks, and conjoin as truths.
        assert_eq!(folds_lit(OpKind::BitAnd, &[t, f]), Some(f));
        assert_eq!(folds_lit(OpKind::BitOr, &[t, f]), Some(t));
    }

    /// The NaN whose lane a true mask shares is a different literal: it is
    /// not a condition `If` will select by, and it does not fold as one.
    #[test]
    fn a_nan_is_not_a_mask_to_the_folder() {
        let nan = Lit::Real(f32::from_bits(u32::MAX));
        assert_eq!(
            folds_lit(OpKind::If, &[nan, Lit::Real(7.0), Lit::Real(9.0)]),
            None
        );
    }
}
