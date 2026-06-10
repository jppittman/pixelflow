//! Arena-allocated expression storage.
//!
//! [`ExprArena`] stores expression nodes in a flat `Vec<ExprNode>`, indexed by
//! [`ExprId`] (a 4-byte Copy handle). This eliminates per-node Arc overhead and
//! gives O(1) `len()` for node counting.
//!
//! The arena is append-only. [`ExprArena::clear`] truncates without deallocating,
//! ready for reuse.

use alloc::vec::Vec;
use core::fmt;

use crate::kind::OpKind;

// ───────────────────────────────────────── ExprId ─────────────────────────────

/// Index into an [`ExprArena`]. Copy, 4 bytes, no refcount.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ExprId(pub u32);

// ───────────────────────────────────────── ExprNode ───────────────────────────

/// A single expression node stored in the arena.
///
/// Layout is kept tight: the static assertion below guarantees <= 16 bytes.
#[derive(Clone, Debug, PartialEq)]
pub enum ExprNode {
    Var(u8),
    Const(f32),
    Param(u8),
    Unary(OpKind, ExprId),
    Binary(OpKind, ExprId, ExprId),
    Ternary(OpKind, ExprId, ExprId, ExprId),
    /// N-ary node. Children live in `ExprArena::nary_children[start..start+len]`.
    Nary(OpKind, u32, u16),
}

const _: () = assert!(
    core::mem::size_of::<ExprNode>() <= 16,
    "ExprNode must fit in 16 bytes"
);

// ───────────────────────────────────── ExprChildren ──────────────────────────

/// Iterator over the child [`ExprId`]s of a node.
pub enum ExprChildren<'a> {
    Zero,
    One(ExprId),
    Two(ExprId, ExprId),
    Three(ExprId, ExprId, ExprId),
    Nary(&'a [ExprId]),
}

impl<'a> Iterator for ExprChildren<'a> {
    type Item = ExprId;

    fn next(&mut self) -> Option<ExprId> {
        match self {
            Self::Zero => None,
            Self::One(id) => {
                let id = *id;
                *self = Self::Zero;
                Some(id)
            }
            Self::Two(a, b) => {
                let a = *a;
                let b = *b;
                *self = Self::One(b);
                Some(a)
            }
            Self::Three(a, b, c) => {
                let a = *a;
                let b = *b;
                let c = *c;
                *self = Self::Two(b, c);
                Some(a)
            }
            Self::Nary(slice) => {
                if slice.is_empty() {
                    None
                } else {
                    let first = slice[0];
                    *self = Self::Nary(&slice[1..]);
                    Some(first)
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match self {
            Self::Zero => 0,
            Self::One(_) => 1,
            Self::Two(_, _) => 2,
            Self::Three(_, _, _) => 3,
            Self::Nary(s) => s.len(),
        };
        (n, Some(n))
    }
}

impl ExactSizeIterator for ExprChildren<'_> {}

// ───────────────────────────────────── ExprArena ─────────────────────────────

/// Arena-allocated expression storage. Append-only, O(1) drop.
#[derive(Clone)]
pub struct ExprArena {
    nodes: Vec<ExprNode>,
    nary_children: Vec<ExprId>,
}

impl ExprArena {
    /// Create an empty arena.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            nary_children: Vec::new(),
        }
    }

    /// Create an arena pre-allocated for `n` nodes.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(n),
            nary_children: Vec::new(),
        }
    }

    /// Truncate to zero nodes without deallocating backing storage.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.nary_children.clear();
    }

    /// Number of nodes in the arena. This is the O(1) node count.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the arena contains no nodes.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    // ───────────────────── push helpers ──────────────────────

    fn push_node(&mut self, node: ExprNode) -> ExprId {
        let id = ExprId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    /// Push a `Var(i)` node.
    pub fn push_var(&mut self, i: u8) -> ExprId {
        self.push_node(ExprNode::Var(i))
    }

    /// Push a `Const(v)` node.
    pub fn push_const(&mut self, v: f32) -> ExprId {
        self.push_node(ExprNode::Const(v))
    }

    /// Push a `Param(i)` node.
    pub fn push_param(&mut self, i: u8) -> ExprId {
        self.push_node(ExprNode::Param(i))
    }

    /// Push a unary operation node.
    pub fn push_unary(&mut self, op: OpKind, child: ExprId) -> ExprId {
        self.push_node(ExprNode::Unary(op, child))
    }

    /// Push a binary operation node.
    pub fn push_binary(&mut self, op: OpKind, a: ExprId, b: ExprId) -> ExprId {
        self.push_node(ExprNode::Binary(op, a, b))
    }

    /// Push a ternary operation node.
    pub fn push_ternary(&mut self, op: OpKind, a: ExprId, b: ExprId, c: ExprId) -> ExprId {
        self.push_node(ExprNode::Ternary(op, a, b, c))
    }

    /// Push an N-ary operation node. Children are copied into the internal slab.
    ///
    /// # Panics
    ///
    /// Panics if `children.len()` exceeds `u16::MAX`.
    pub fn push_nary(&mut self, op: OpKind, children: &[ExprId]) -> ExprId {
        assert!(
            children.len() <= u16::MAX as usize,
            "push_nary: {} children exceeds u16::MAX",
            children.len()
        );
        let start = self.nary_children.len() as u32;
        let len = children.len() as u16;
        self.nary_children.extend_from_slice(children);
        self.push_node(ExprNode::Nary(op, start, len))
    }

    // ───────────────────── raw access (serialization) ───────

    /// Raw slice of all nodes in the arena.
    #[inline]
    #[must_use]
    pub fn nodes_raw(&self) -> &[ExprNode] {
        &self.nodes
    }

    /// Raw slice of the nary-children slab.
    #[inline]
    #[must_use]
    pub fn nary_children_raw(&self) -> &[ExprId] {
        &self.nary_children
    }

    /// Reconstruct an arena from raw parts.
    ///
    /// # Safety contract (logical, not `unsafe`)
    ///
    /// The caller must ensure that every `ExprId` referenced by nodes in
    /// `nodes` is in-bounds, and that `Nary` start/len pairs index validly
    /// into `nary_children`. Violating this will cause panics on access,
    /// not UB.
    #[must_use]
    pub fn from_raw(nodes: Vec<ExprNode>, nary_children: Vec<ExprId>) -> Self {
        Self {
            nodes,
            nary_children,
        }
    }

    // ───────────────────── access ────────────────────────────

    /// Get the node at `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of bounds.
    #[inline]
    pub fn node(&self, id: ExprId) -> &ExprNode {
        &self.nodes[id.0 as usize]
    }

    /// Get the N-ary children slice for a `Nary(_, start, len)` node.
    ///
    /// # Panics
    ///
    /// Panics if `start + len` exceeds the internal nary_children buffer.
    #[inline]
    pub fn nary_children_slice(&self, start: u32, len: u16) -> &[ExprId] {
        let s = start as usize;
        let l = len as usize;
        &self.nary_children[s..s + l]
    }

    /// Get the [`OpKind`] of the node at `id`.
    ///
    /// Leaf nodes map to: `Var -> OpKind::Var`, `Const/Param -> OpKind::Const`.
    #[inline]
    pub fn kind(&self, id: ExprId) -> OpKind {
        match &self.nodes[id.0 as usize] {
            ExprNode::Var(_) => OpKind::Var,
            ExprNode::Const(_) | ExprNode::Param(_) => OpKind::Const,
            ExprNode::Unary(op, _) => *op,
            ExprNode::Binary(op, _, _) => *op,
            ExprNode::Ternary(op, _, _, _) => *op,
            ExprNode::Nary(op, _, _) => *op,
        }
    }

    /// Iterate over the child [`ExprId`]s of the node at `id`.
    #[inline]
    pub fn children(&self, id: ExprId) -> ExprChildren<'_> {
        match &self.nodes[id.0 as usize] {
            ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Param(_) => ExprChildren::Zero,
            ExprNode::Unary(_, a) => ExprChildren::One(*a),
            ExprNode::Binary(_, a, b) => ExprChildren::Two(*a, *b),
            ExprNode::Ternary(_, a, b, c) => ExprChildren::Three(*a, *b, *c),
            ExprNode::Nary(_, start, len) => {
                let s = *start as usize;
                let l = *len as usize;
                ExprChildren::Nary(&self.nary_children[s..s + l])
            }
        }
    }

    // ───────────────────── traversal ─────────────────────────

    /// Compute the depth of the subtree rooted at `root` (iterative).
    #[must_use]
    pub fn depth(&self, root: ExprId) -> usize {
        let mut stack: Vec<(ExprId, usize)> = Vec::new();
        stack.push((root, 1));
        let mut max_depth: usize = 0;

        while let Some((id, d)) = stack.pop() {
            match &self.nodes[id.0 as usize] {
                ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Param(_) => {
                    max_depth = max_depth.max(d);
                }
                ExprNode::Unary(_, a) => {
                    stack.push((*a, d + 1));
                }
                ExprNode::Binary(_, a, b) => {
                    stack.push((*a, d + 1));
                    stack.push((*b, d + 1));
                }
                ExprNode::Ternary(_, a, b, c) => {
                    stack.push((*a, d + 1));
                    stack.push((*b, d + 1));
                    stack.push((*c, d + 1));
                }
                ExprNode::Nary(_, start, len) => {
                    let s = *start as usize;
                    let l = *len as usize;
                    if l == 0 {
                        max_depth = max_depth.max(d);
                    } else {
                        for child in &self.nary_children[s..s + l] {
                            stack.push((*child, d + 1));
                        }
                    }
                }
            }
        }
        max_depth
    }

    /// Returns `true` if the subtree rooted at `root` contains at least one `Var` node.
    #[must_use]
    pub fn has_var(&self, root: ExprId) -> bool {
        let mut stack: Vec<ExprId> = Vec::new();
        stack.push(root);

        while let Some(id) = stack.pop() {
            match &self.nodes[id.0 as usize] {
                ExprNode::Var(_) => return true,
                ExprNode::Const(_) | ExprNode::Param(_) => {}
                ExprNode::Unary(_, a) => stack.push(*a),
                ExprNode::Binary(_, a, b) => {
                    stack.push(*a);
                    stack.push(*b);
                }
                ExprNode::Ternary(_, a, b, c) => {
                    stack.push(*a);
                    stack.push(*b);
                    stack.push(*c);
                }
                ExprNode::Nary(_, start, len) => {
                    let s = *start as usize;
                    let l = *len as usize;
                    for child in &self.nary_children[s..s + l] {
                        stack.push(*child);
                    }
                }
            }
        }
        false
    }

    /// Returns `true` if the subtree contains degenerate subexpressions:
    /// NaN/Inf constants, `recip(0)`, `div(_, 0)`.
    #[must_use]
    pub fn has_degenerate(&self, root: ExprId) -> bool {
        let mut stack: Vec<ExprId> = vec![root];

        while let Some(id) = stack.pop() {
            match &self.nodes[id.0 as usize] {
                ExprNode::Const(v) if !v.is_finite() => return true,
                ExprNode::Unary(OpKind::Recip, a) => {
                    if matches!(self.nodes[a.0 as usize], ExprNode::Const(v) if v == 0.0) {
                        return true;
                    }
                    stack.push(*a);
                }
                ExprNode::Binary(OpKind::Div, a, b) => {
                    if matches!(self.nodes[b.0 as usize], ExprNode::Const(v) if v == 0.0) {
                        return true;
                    }
                    stack.push(*a);
                    stack.push(*b);
                }
                ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Param(_) => {}
                ExprNode::Unary(_, a) => stack.push(*a),
                ExprNode::Binary(_, a, b) => {
                    stack.push(*a);
                    stack.push(*b);
                }
                ExprNode::Ternary(_, a, b, c) => {
                    stack.push(*a);
                    stack.push(*b);
                    stack.push(*c);
                }
                ExprNode::Nary(_, start, len) => {
                    let s = *start as usize;
                    let l = *len as usize;
                    for child in &self.nary_children[s..s + l] {
                        stack.push(*child);
                    }
                }
            }
        }
        false
    }

    /// Count total nodes reachable from `root` (iterative).
    ///
    /// Note: if the DAG shares subtrees (same ExprId referenced multiple times),
    /// shared nodes are counted once per reference. This matches `Expr::node_count`
    /// behavior on Arc trees (where shared subtrees are traversed per reference).
    #[must_use]
    pub fn node_count_subtree(&self, root: ExprId) -> usize {
        let mut stack: Vec<ExprId> = Vec::new();
        stack.push(root);
        let mut count: usize = 0;

        while let Some(id) = stack.pop() {
            count += 1;
            match &self.nodes[id.0 as usize] {
                ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Param(_) => {}
                ExprNode::Unary(_, a) => stack.push(*a),
                ExprNode::Binary(_, a, b) => {
                    stack.push(*a);
                    stack.push(*b);
                }
                ExprNode::Ternary(_, a, b, c) => {
                    stack.push(*a);
                    stack.push(*b);
                    stack.push(*c);
                }
                ExprNode::Nary(_, start, len) => {
                    let s = *start as usize;
                    let l = *len as usize;
                    for child in &self.nary_children[s..s + l] {
                        stack.push(*child);
                    }
                }
            }
        }
        count
    }

    /// Replace every `Param(i)` node with `Const(params[i])`.
    ///
    /// Returns the new root [`ExprId`] in the **same** arena. Old nodes become
    /// unreachable garbage — that is fine for an append-only arena.
    ///
    /// # Panics
    ///
    /// Panics if any `Param(i)` has `i >= params.len()`.
    pub fn substitute_params(&mut self, root: ExprId, params: &[f32]) -> ExprId {
        // Iterative post-order: map old ExprId -> new ExprId.
        // We use a Vec as a dense map since IDs are contiguous 0..n.
        enum Task {
            Descend(ExprId),
            Emit(ExprId),
        }

        // We'll build a mapping: old_id -> new_id.
        // Initialize with sentinel values.
        let old_len = self.nodes.len();
        let mut id_map: Vec<Option<ExprId>> = Vec::new();
        id_map.resize(old_len, None);

        let mut work: Vec<Task> = vec![Task::Descend(root)];

        while let Some(task) = work.pop() {
            match task {
                Task::Descend(id) => {
                    // If already mapped (shared subtree), skip.
                    if id_map[id.0 as usize].is_some() {
                        continue;
                    }
                    work.push(Task::Emit(id));
                    match &self.nodes[id.0 as usize] {
                        ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Param(_) => {}
                        ExprNode::Unary(_, a) => {
                            work.push(Task::Descend(*a));
                        }
                        ExprNode::Binary(_, a, b) => {
                            work.push(Task::Descend(*b));
                            work.push(Task::Descend(*a));
                        }
                        ExprNode::Ternary(_, a, b, c) => {
                            work.push(Task::Descend(*c));
                            work.push(Task::Descend(*b));
                            work.push(Task::Descend(*a));
                        }
                        ExprNode::Nary(_, start, len) => {
                            let s = *start as usize;
                            let l = *len as usize;
                            for child in self.nary_children[s..s + l].iter().rev() {
                                work.push(Task::Descend(*child));
                            }
                        }
                    }
                }
                Task::Emit(id) => {
                    // Skip if already emitted (can happen with shared subtrees).
                    if id_map[id.0 as usize].is_some() {
                        continue;
                    }
                    let new_id = match self.nodes[id.0 as usize].clone() {
                        ExprNode::Param(i) => {
                            let idx = i as usize;
                            assert!(
                                idx < params.len(),
                                "substitute_params: param index {} out of range (have {} params)",
                                idx,
                                params.len()
                            );
                            self.push_const(params[idx])
                        }
                        ExprNode::Var(i) => self.push_var(i),
                        ExprNode::Const(v) => self.push_const(v),
                        ExprNode::Unary(op, a) => {
                            let na = id_map[a.0 as usize]
                                .expect("substitute_params: child not yet mapped for Unary");
                            self.push_unary(op, na)
                        }
                        ExprNode::Binary(op, a, b) => {
                            let na = id_map[a.0 as usize]
                                .expect("substitute_params: child a not yet mapped for Binary");
                            let nb = id_map[b.0 as usize]
                                .expect("substitute_params: child b not yet mapped for Binary");
                            self.push_binary(op, na, nb)
                        }
                        ExprNode::Ternary(op, a, b, c) => {
                            let na = id_map[a.0 as usize]
                                .expect("substitute_params: child a not yet mapped for Ternary");
                            let nb = id_map[b.0 as usize]
                                .expect("substitute_params: child b not yet mapped for Ternary");
                            let nc = id_map[c.0 as usize]
                                .expect("substitute_params: child c not yet mapped for Ternary");
                            self.push_ternary(op, na, nb, nc)
                        }
                        ExprNode::Nary(op, start, len) => {
                            let s = start as usize;
                            let l = len as usize;
                            let child_ids: Vec<ExprId> = self.nary_children[s..s + l]
                                .iter()
                                .map(|old_child| {
                                    id_map[old_child.0 as usize]
                                        .expect("substitute_params: nary child not yet mapped")
                                })
                                .collect();
                            self.push_nary(op, &child_ids)
                        }
                    };
                    id_map[id.0 as usize] = Some(new_id);
                }
            }
        }

        id_map[root.0 as usize].expect("substitute_params: root was never mapped")
    }

    // ───────────────────── display ───────────────────────────

    /// Format the subtree rooted at `root` as an S-expression, matching the
    /// [`Expr`] display format.
    pub fn fmt_expr(&self, root: ExprId, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        enum Task {
            Visit(ExprId),
            WriteStr(&'static str),
        }

        let mut stack: Vec<Task> = vec![Task::Visit(root)];

        while let Some(task) = stack.pop() {
            match task {
                Task::WriteStr(s) => f.write_str(s)?,
                Task::Visit(id) => match &self.nodes[id.0 as usize] {
                    ExprNode::Var(i) => write!(f, "Var({})", i)?,
                    ExprNode::Const(v) => write!(f, "Const({})", v)?,
                    ExprNode::Param(i) => write!(f, "Param({})", i)?,
                    ExprNode::Unary(op, a) => {
                        stack.push(Task::WriteStr(")"));
                        stack.push(Task::Visit(*a));
                        f.write_str(op.name())?;
                        f.write_str("(")?;
                    }
                    ExprNode::Binary(op, a, b) => {
                        stack.push(Task::WriteStr(")"));
                        stack.push(Task::Visit(*b));
                        stack.push(Task::WriteStr(", "));
                        stack.push(Task::Visit(*a));
                        f.write_str(op.name())?;
                        f.write_str("(")?;
                    }
                    ExprNode::Ternary(op, a, b, c) => {
                        stack.push(Task::WriteStr(")"));
                        stack.push(Task::Visit(*c));
                        stack.push(Task::WriteStr(", "));
                        stack.push(Task::Visit(*b));
                        stack.push(Task::WriteStr(", "));
                        stack.push(Task::Visit(*a));
                        f.write_str(op.name())?;
                        f.write_str("(")?;
                    }
                    ExprNode::Nary(op, start, len) => {
                        let s = *start as usize;
                        let l = *len as usize;
                        stack.push(Task::WriteStr(")"));
                        for (i, child) in self.nary_children[s..s + l].iter().enumerate().rev() {
                            stack.push(Task::Visit(*child));
                            if i > 0 {
                                stack.push(Task::WriteStr(", "));
                            }
                        }
                        f.write_str(op.name())?;
                        f.write_str("(")?;
                    }
                },
            }
        }
        Ok(())
    }

    /// Return a [`Display`]-able wrapper for the subtree rooted at `root`.
    #[must_use]
    pub fn display(&self, root: ExprId) -> DisplayExpr<'_> {
        DisplayExpr { arena: self, root }
    }

    /// Compare two subtrees for structural equality without allocating `Expr` trees.
    ///
    /// `self[a]` is compared against `other[b]` node-by-node in lockstep using an
    /// iterative work stack. Both subtrees may live in the same arena (pass `self`
    /// for both `self` and `other`) or in different arenas.
    ///
    /// Constant nodes are compared by exact bit equality (same behaviour as
    /// [`Expr`]'s `PartialEq`). Var and Param indices are compared by value.
    #[must_use]
    pub fn subtree_eq(&self, a: ExprId, other: &ExprArena, b: ExprId) -> bool {
        // Stack of (self-id, other-id) pairs still to be compared.
        let mut stack: Vec<(ExprId, ExprId)> = Vec::with_capacity(16);
        stack.push((a, b));

        while let Some((s_id, o_id)) = stack.pop() {
            let s_node = &self.nodes[s_id.0 as usize];
            let o_node = &other.nodes[o_id.0 as usize];

            match (s_node, o_node) {
                (ExprNode::Var(si), ExprNode::Var(oi)) => {
                    if si != oi {
                        return false;
                    }
                }
                (ExprNode::Const(sv), ExprNode::Const(ov)) => {
                    // Bit-exact comparison matches Expr's PartialEq behaviour.
                    if sv.to_bits() != ov.to_bits() {
                        return false;
                    }
                }
                (ExprNode::Param(si), ExprNode::Param(oi)) => {
                    if si != oi {
                        return false;
                    }
                }
                (ExprNode::Unary(s_op, s_a), ExprNode::Unary(o_op, o_a)) => {
                    if s_op != o_op {
                        return false;
                    }
                    stack.push((*s_a, *o_a));
                }
                (ExprNode::Binary(s_op, s_a, s_b), ExprNode::Binary(o_op, o_a, o_b)) => {
                    if s_op != o_op {
                        return false;
                    }
                    stack.push((*s_a, *o_a));
                    stack.push((*s_b, *o_b));
                }
                (
                    ExprNode::Ternary(s_op, s_a, s_b, s_c),
                    ExprNode::Ternary(o_op, o_a, o_b, o_c),
                ) => {
                    if s_op != o_op {
                        return false;
                    }
                    stack.push((*s_a, *o_a));
                    stack.push((*s_b, *o_b));
                    stack.push((*s_c, *o_c));
                }
                (ExprNode::Nary(s_op, s_start, s_len), ExprNode::Nary(o_op, o_start, o_len)) => {
                    if s_op != o_op || s_len != o_len {
                        return false;
                    }
                    let ss = *s_start as usize;
                    let os = *o_start as usize;
                    let len = *s_len as usize;
                    for i in 0..len {
                        stack.push((self.nary_children[ss + i], other.nary_children[os + i]));
                    }
                }
                // Different node variants — structurally unequal.
                _ => return false,
            }
        }

        true
    }
}

// ───────────────────────────────────── DisplayExpr ───────────────────────────

/// Wrapper that implements [`fmt::Display`] for an arena subtree.
pub struct DisplayExpr<'a> {
    arena: &'a ExprArena,
    root: ExprId,
}

impl fmt::Display for DisplayExpr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.arena.fmt_expr(self.root, f)
    }
}

// ───────────────────────────────────── Tests ─────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    // 1. test_push_and_access
    #[test]
    fn test_push_and_access() {
        let mut arena = ExprArena::new();

        let v = arena.push_var(0);
        assert_eq!(arena.kind(v), OpKind::Var);
        assert_eq!(arena.children(v).count(), 0);

        let c = arena.push_const(core::f32::consts::PI);
        assert_eq!(arena.kind(c), OpKind::Const);
        assert_eq!(arena.children(c).count(), 0);

        let p = arena.push_param(1);
        assert_eq!(arena.kind(p), OpKind::Const); // Param maps to Const kind
        assert_eq!(arena.children(p).count(), 0);

        let u = arena.push_unary(OpKind::Neg, v);
        assert_eq!(arena.kind(u), OpKind::Neg);
        let u_children: Vec<ExprId> = arena.children(u).collect();
        assert_eq!(u_children, vec![v]);

        let b = arena.push_binary(OpKind::Add, v, c);
        assert_eq!(arena.kind(b), OpKind::Add);
        let b_children: Vec<ExprId> = arena.children(b).collect();
        assert_eq!(b_children, vec![v, c]);

        let t = arena.push_ternary(OpKind::MulAdd, v, c, p);
        assert_eq!(arena.kind(t), OpKind::MulAdd);
        let t_children: Vec<ExprId> = arena.children(t).collect();
        assert_eq!(t_children, vec![v, c, p]);
    }

    // 2. test_node_count
    #[test]
    fn test_node_count() {
        let mut arena = ExprArena::new();
        let v0 = arena.push_var(0);
        let c1 = arena.push_const(1.0);
        let root = arena.push_binary(OpKind::Add, v0, c1);
        assert_eq!(arena.len(), 3);
        assert_eq!(arena.node_count_subtree(root), 3);

        let mut arena2 = ExprArena::new();
        let a0 = arena2.push_var(0);
        let a1 = arena2.push_var(1);
        let add = arena2.push_binary(OpKind::Add, a0, a1);
        let c2 = arena2.push_const(2.0);
        let root2 = arena2.push_binary(OpKind::Mul, add, c2);
        assert_eq!(arena2.len(), 5);
        assert_eq!(arena2.node_count_subtree(root2), 5);
    }

    // 3. test_depth
    #[test]
    fn test_depth() {
        let mut arena = ExprArena::new();
        let v0 = arena.push_var(0);
        let v1 = arena.push_var(1);
        let c3 = arena.push_const(3.0);
        let mul = arena.push_binary(OpKind::Mul, v1, c3);
        let root = arena.push_binary(OpKind::Add, v0, mul);
        assert_eq!(arena.depth(root), 3);
    }

    // 4. test_has_var
    #[test]
    fn test_has_var() {
        let mut arena1 = ExprArena::new();
        let v0 = arena1.push_var(0);
        let c1 = arena1.push_const(1.0);
        let root1 = arena1.push_binary(OpKind::Add, v0, c1);
        assert!(arena1.has_var(root1));

        let mut arena2 = ExprArena::new();
        let c1 = arena2.push_const(1.0);
        let c2 = arena2.push_const(2.0);
        let root2 = arena2.push_binary(OpKind::Add, c1, c2);
        assert!(!arena2.has_var(root2));
    }

    // 5. test_has_degenerate
    #[test]
    fn test_has_degenerate() {
        let mut arena1 = ExprArena::new();
        let root1 = arena1.push_const(f32::NAN);
        assert!(arena1.has_degenerate(root1));

        let mut arena2 = ExprArena::new();
        let root2 = arena2.push_const(f32::INFINITY);
        assert!(arena2.has_degenerate(root2));

        let mut arena3 = ExprArena::new();
        let v0 = arena3.push_var(0);
        let c0 = arena3.push_const(0.0);
        let root3 = arena3.push_binary(OpKind::Div, v0, c0);
        assert!(arena3.has_degenerate(root3));

        let mut arena4 = ExprArena::new();
        let c0 = arena4.push_const(0.0);
        let root4 = arena4.push_unary(OpKind::Recip, c0);
        assert!(arena4.has_degenerate(root4));

        let mut arena5 = ExprArena::new();
        let v0 = arena5.push_var(0);
        let c1 = arena5.push_const(1.0);
        let root5 = arena5.push_binary(OpKind::Add, v0, c1);
        assert!(!arena5.has_degenerate(root5));
    }

    // 7. test_clear_preserves_capacity
    #[test]
    fn test_clear_preserves_capacity() {
        let mut arena = ExprArena::with_capacity(64);
        let _v = arena.push_var(0);
        let _c = arena.push_const(1.0);
        assert_eq!(arena.len(), 2);

        arena.clear();
        assert_eq!(arena.len(), 0);
        assert!(arena.is_empty());

        // Push again — should work fine, capacity preserved.
        let v2 = arena.push_var(1);
        assert_eq!(v2, ExprId(0));
        assert_eq!(arena.len(), 1);
    }

    // 7. test_substitute_params
    #[test]
    fn test_substitute_params() {
        let mut arena = ExprArena::new();
        let p0 = arena.push_param(0);
        let p1 = arena.push_param(1);
        let root = arena.push_binary(OpKind::Add, p0, p1);

        let new_root = arena.substitute_params(root, &[10.0, 20.0]);

        match arena.node(new_root) {
            ExprNode::Binary(OpKind::Add, a, b) => {
                assert!(matches!(arena.node(*a), ExprNode::Const(v) if (*v - 10.0).abs() < 1e-6));
                assert!(matches!(arena.node(*b), ExprNode::Const(v) if (*v - 20.0).abs() < 1e-6));
            }
            other => panic!("expected Binary(Add, ...), got {:?}", other),
        }
    }

    // 8. test_nary
    #[test]
    fn test_nary() {
        let mut arena = ExprArena::new();
        let v0 = arena.push_var(0);
        let v1 = arena.push_var(1);
        let c = arena.push_const(42.0);

        let tup = arena.push_nary(OpKind::Tuple, &[v0, v1, c]);
        assert_eq!(arena.kind(tup), OpKind::Tuple);

        let children: Vec<ExprId> = arena.children(tup).collect();
        assert_eq!(children, vec![v0, v1, c]);
        assert_eq!(arena.children(tup).len(), 3);
    }

    // 9. test_display
    #[test]
    fn test_display() {
        let mut arena = ExprArena::new();
        let v0 = arena.push_var(0);
        let v1 = arena.push_var(1);
        let c2 = arena.push_const(2.0);
        let root = arena.push_ternary(OpKind::MulAdd, v0, v1, c2);
        // `display` matches the canonical `Expr` S-expression format.
        assert_eq!(
            format!("{}", arena.display(root)),
            "mul_add(Var(0), Var(1), Const(2))"
        );
    }

    #[test]
    fn test_size_of_expr_node() {
        // Compile-time assertion exists above, but also verify at runtime.
        assert!(
            core::mem::size_of::<ExprNode>() <= 16,
            "ExprNode is {} bytes, expected <= 16",
            core::mem::size_of::<ExprNode>()
        );
    }

    #[test]
    fn test_expr_children_exact_size() {
        let mut arena = ExprArena::new();
        let v = arena.push_var(0);
        let c = arena.push_const(1.0);
        let bin = arena.push_binary(OpKind::Add, v, c);

        assert_eq!(arena.children(v).len(), 0);
        assert_eq!(arena.children(bin).len(), 2);
    }
}
