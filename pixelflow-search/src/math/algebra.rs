//! Core algebraic rewrite rules.
//!
//! This module contains:
//! - `InversePair` trait: derives 4 rules from one trait impl (canonicalize, involution, cancellation, annihilation)
//! - Basic algebraic rules: commutativity, identity, annihilator, associativity, idempotent
//! - Distributivity and factoring

use std::marker::PhantomData;

use crate::arena_pat;
use pixelflow_ir::arena::{ExprArena, ExprId};
use crate::egraph::{EClassId, EGraph, ENode, Op, Rewrite, RewriteAction, ops};
use pixelflow_ir::OpKind;


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
    fn base() -> &'static dyn Op;
    /// The inverse operation (Neg, Recip)
    fn inverse() -> &'static dyn Op;
    /// The derived operation (Sub, Div)
    fn derived() -> &'static dyn Op;
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
    fn base() -> &'static dyn Op {
        &ops::Add
    }
    fn inverse() -> &'static dyn Op {
        &ops::Neg
    }
    fn derived() -> &'static dyn Op {
        &ops::Sub
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
    fn base() -> &'static dyn Op {
        &ops::Mul
    }
    fn inverse() -> &'static dyn Op {
        &ops::Recip
    }
    fn derived() -> &'static dyn Op {
        &ops::Div
    }
    fn identity() -> f32 {
        1.0
    }
}

// ============================================================================
// Helper: Check if node matches an operation by kind
// ============================================================================

fn node_matches_op(node: &ENode, op: &dyn Op) -> bool {
    match node {
        ENode::Op { op: node_op, .. } => node_op.kind() == op.kind(),
        _ => false,
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

    fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if !node_matches_op(node, T::derived()) {
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
        Some(arena_pat!(__a, bin T::derived().kind(), (var 0), (var 1)))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin T::base().kind(), (var 0), (un T::inverse().kind(), (var 1))))
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
    fn is_destructive(&self) -> bool {
        true
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if !node_matches_op(node, T::inverse()) {
            return None;
        }

        let children = node.children();
        if children.len() != 1 {
            return None;
        }
        let inner_id = children[0];

        for inner_node in egraph.nodes(inner_id) {
            if node_matches_op(inner_node, T::inverse()) {
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
        let inv = T::inverse().kind();
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
    fn is_destructive(&self) -> bool {
        true
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if !node_matches_op(node, T::derived()) {
            return None;
        }
        let (numerator, canceller) = node.binary_operands()?;

        for inner_node in egraph.nodes(numerator) {
            if node_matches_op(inner_node, T::base()) {
                if let Some((a, b)) = inner_node.binary_operands() {
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
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin T::derived().kind(), (bin T::base().kind(), (var 0), (var 1)), (var 1)))
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

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if !node_matches_op(node, T::base()) {
            return None;
        }
        let (a, b) = node.binary_operands()?;

        // x ⊕ inv(x) → identity
        for node_b in egraph.nodes(b) {
            if node_matches_op(node_b, T::inverse()) {
                if let Some(&inner) = node_b.children().first() {
                    if egraph.find(inner) == egraph.find(a) {
                        return Some(RewriteAction::Create(ENode::constant(T::identity())));
                    }
                }
            }
        }

        // inv(x) ⊕ x → identity
        for node_a in egraph.nodes(a) {
            if node_matches_op(node_a, T::inverse()) {
                if let Some(&inner) = node_a.children().first() {
                    if egraph.find(inner) == egraph.find(b) {
                        return Some(RewriteAction::Create(ENode::constant(T::identity())));
                    }
                }
            }
        }

        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin T::base().kind(), (var 0), (un T::inverse().kind(), (var 0))))
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
    op: &'static dyn Op,
}

impl Associative {
    pub fn new(op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Associative {
    fn name(&self) -> &str {
        "associative"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.kind() != self.op.kind() {
            return None;
        }

        let (left, right) = node.binary_operands()?;

        for child in egraph.nodes(left) {
            if let Some(child_op) = child.op() {
                if child_op.kind() == self.op.kind() {
                    if let Some((a, b)) = child.binary_operands() {
                        return Some(RewriteAction::Associate {
                            op: self.op,
                            a,
                            b,
                            c: right,
                        });
                    }
                }
            }
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {

        // Op(Op(V0, V1), V2)
        let k = self.op.kind();
        Some(arena_pat!(__a, bin k, (bin k, (var 0), (var 1)), (var 2)))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {

        // Op(V0, Op(V1, V2))
        let k = self.op.kind();
        Some(arena_pat!(__a, bin k, (var 0), (bin k, (var 1), (var 2))))
    }
}

/// Reverse associativity: a op (b op c) → (a op b) op c
pub struct ReverseAssociative {
    op: &'static dyn Op,
}

impl ReverseAssociative {
    pub fn new(op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for ReverseAssociative {
    fn name(&self) -> &str {
        "reverse-associative"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.kind() != self.op.kind() {
            return None;
        }

        let (left, right) = node.binary_operands()?;

        // Check if the right child has a node with the same op
        for child in egraph.nodes(right) {
            if let Some(child_op) = child.op() {
                if child_op.kind() == self.op.kind() {
                    if let Some((b, c)) = child.binary_operands() {
                        return Some(RewriteAction::ReverseAssociate {
                            op: self.op,
                            a: left,
                            b,
                            c,
                        });
                    }
                }
            }
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {

        // Op(V0, Op(V1, V2))
        let k = self.op.kind();
        Some(arena_pat!(__a, bin k, (var 0), (bin k, (var 1), (var 2))))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {

        // Op(Op(V0, V1), V2)
        let k = self.op.kind();
        Some(arena_pat!(__a, bin k, (bin k, (var 0), (var 1)), (var 2)))
    }
}

/// Commutativity: a op b → b op a
pub struct Commutative {
    op: &'static dyn Op,
}

impl Commutative {
    pub fn new(op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Commutative {
    fn name(&self) -> &str {
        "commutative"
    }

    fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.kind() != self.op.kind() {
            return None;
        }

        let (a, b) = node.binary_operands()?;
        if a == b {
            return None;
        }

        let swapped = ENode::Op {
            op: self.op,
            children: vec![b, a],
        };
        Some(RewriteAction::Create(swapped))
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin self.op.kind(), (var 0), (var 1)))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin self.op.kind(), (var 1), (var 0)))
    }
}

/// Distributivity: A * (B + C) → A*B + A*C
pub struct Distributive {
    outer: &'static dyn Op,
    inner: &'static dyn Op,
}

impl Distributive {
    pub fn new(outer: &'static dyn Op, inner: &'static dyn Op) -> Box<Self> {
        Box::new(Self { outer, inner })
    }
}

impl Rewrite for Distributive {
    fn name(&self) -> &str {
        "distribute"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.kind() != self.outer.kind() {
            return None;
        }

        let (a, other) = node.binary_operands()?;

        for child_node in egraph.nodes(other) {
            if let Some(child_op) = child_node.op() {
                if child_op.kind() == self.inner.kind() {
                    if let Some((b, c)) = child_node.binary_operands() {
                        return Some(RewriteAction::Distribute {
                            outer: self.outer,
                            inner: self.inner,
                            a,
                            b,
                            c,
                        });
                    }
                }
            }
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin self.outer.kind(), (var 0), (bin self.inner.kind(), (var 1), (var 2))))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {

        // Inner(Outer(V0, V1), Outer(V0, V2))
        let ok = self.outer.kind();
        let ik = self.inner.kind();
        Some(arena_pat!(__a, bin ik, (bin ok, (var 0), (var 1)), (bin ok, (var 0), (var 2))))
    }
}

/// Factoring: A*B + A*C → A * (B + C)
pub struct Factor {
    outer: &'static dyn Op,
    inner: &'static dyn Op,
}

impl Factor {
    pub fn new(outer: &'static dyn Op, inner: &'static dyn Op) -> Box<Self> {
        Box::new(Self { outer, inner })
    }
}

impl Rewrite for Factor {
    fn name(&self) -> &str {
        "factor"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.kind() != self.outer.kind() {
            return None;
        }

        let (left, right) = node.binary_operands()?;

        for l_node in egraph.nodes(left) {
            let l_op = l_node.op()?;
            if l_op.kind() != self.inner.kind() {
                continue;
            }
            let (la, lb) = l_node.binary_operands()?;

            for r_node in egraph.nodes(right) {
                let r_op = r_node.op()?;
                if r_op.kind() != self.inner.kind() {
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
        let ok = self.outer.kind();
        let ik = self.inner.kind();
        Some(arena_pat!(__a, bin ok, (bin ik, (var 0), (var 1)), (bin ik, (var 0), (var 2))))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin self.inner.kind(), (var 0), (bin self.outer.kind(), (var 1), (var 2))))
    }
}

/// Identity: x op identity → x
pub struct Identity {
    op: &'static dyn Op,
}

impl Identity {
    pub fn new(op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Identity {
    fn name(&self) -> &str {
        "identity"
    }
    fn is_destructive(&self) -> bool {
        true
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.kind() != self.op.kind() {
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
        Some(arena_pat!(__a, bin self.op.kind(), (var 0), (cst id_val)))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, var 0))
    }
}

/// Annihilator: x op annihilator → annihilator
pub struct Annihilator {
    op: &'static dyn Op,
}

impl Annihilator {
    pub fn new(op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Annihilator {
    fn name(&self) -> &str {
        "annihilator"
    }
    fn is_destructive(&self) -> bool {
        true
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.kind() != self.op.kind() {
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
        Some(arena_pat!(__a, bin self.op.kind(), (var 0), (cst ann)))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {

        // Const(annihilator)
        let ann = self.op.annihilator()?;
        Some(arena_pat!(__a, cst ann))
    }
}

/// Idempotence: x op x → x
pub struct Idempotent {
    op: &'static dyn Op,
}

impl Idempotent {
    pub fn new(op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { op })
    }
}

impl Rewrite for Idempotent {
    fn name(&self) -> &str {
        "idempotent"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.kind() != self.op.kind() {
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
        Some(arena_pat!(__a, bin self.op.kind(), (var 0), (var 0)))
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

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: Add(a, b) where a == b
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != ops::Add.kind() {
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

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: Mul(2, a) or Mul(a, 2)
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != ops::Mul.kind() {
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
    fn is_destructive(&self) -> bool {
        true
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };

        let kind = op.kind();

        // Collect constant values from all children
        let mut args = Vec::with_capacity(children.len());
        for &child_id in children {
            // Check if this e-class contains a constant
            let mut found_const = None;
            for child_node in egraph.nodes(child_id) {
                if let Some(val) = child_node.as_f32() {
                    found_const = Some(val);
                    break;
                }
            }
            args.push(found_const?);
        }

        // Evaluate based on arity
        let result = match args.len() {
            1 => kind.eval_unary(args[0])?,
            2 => kind.eval_binary(args[0], args[1])?,
            3 => kind.eval_ternary(args[0], args[1], args[2])?,
            _ => return None,
        };

        // Don't fold NaN or infinity - they can cause issues
        if !result.is_finite() {
            return None;
        }

        Some(RewriteAction::Create(ENode::constant(result)))
    }
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
        Commutative::new(&ops::Add),
        Commutative::new(&ops::Mul),
        Commutative::new(&ops::Min),
        Commutative::new(&ops::Max),
        // Identity elements: x + 0 → x, x * 1 → x
        Identity::new(&ops::Add),
        Identity::new(&ops::Mul),
        // Annihilators: x * 0 → 0
        Annihilator::new(&ops::Mul),
        // Idempotent: min(x,x) → x, max(x,x) → x
        Idempotent::new(&ops::Min),
        Idempotent::new(&ops::Max),
        // Distributivity: a * (b + c) → a*b + a*c
        Distributive::new(&ops::Mul, &ops::Add),
        // Factoring: a*b + a*c → a * (b + c)
        Factor::new(&ops::Add, &ops::Mul),
        // Doubling/Halving: a + a ↔ 2 * a
        Doubling::new(),
        Halving::new(),
        // Associativity (L→R): (a op b) op c → a op (b op c)
        Associative::new(&ops::Add),
        Associative::new(&ops::Mul),
        Associative::new(&ops::Min),
        Associative::new(&ops::Max),
        // Reverse associativity (R→L): a op (b op c) → (a op b) op c
        ReverseAssociative::new(&ops::Add),
        ReverseAssociative::new(&ops::Mul),
        ReverseAssociative::new(&ops::Min),
        ReverseAssociative::new(&ops::Max),
    ]
}

/// All algebra rules combined.
pub fn algebra_rules() -> Vec<Box<dyn Rewrite>> {
    let mut rules = inverse_pair_rules();
    rules.extend(basic_algebra_rules());
    rules
}
