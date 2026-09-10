//! Arena-allocated expression storage.
//!
//! [`ExprArena`] stores expression nodes in a flat `Vec<ExprNode>`, indexed by
//! [`ExprId`] (a 4-byte Copy handle). This eliminates per-node Arc overhead and
//! gives O(1) `len()` for node counting.
//!
//! The arena is append-only. [`ExprArena::clear`] truncates without deallocating,
//! ready for reuse.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::fold::Fold;
use crate::kernel::Scalar;
use crate::key::KernelKey;
use crate::kind::OpKind;

/// Coordinate axes a lattice has, and so the coordinate `Var` indices: `X = 0`,
/// `Y = 1`.
///
/// There were four. Z and W had extent 1 in every production call — an axis
/// that never varies is not an axis — so they left the language and the
/// scalars they carried became [`UniformDecl`]s
/// (docs/plans/2026-09-06-lattice-is-the-index.md).
pub const COORD_AXES: usize = 2;

/// The `Var` indices Z and W had. Reserved, never reissued: a reduction
/// binder taking one of them would make an arena written before the change
/// read back as a different program.
pub(crate) const RETIRED_COORD_AXES: [u8; 2] = [2, 3];

/// The first `Var` index a reduction binder takes.
///
/// Binder indices are dense from here, and *here* is past the reserved
/// [`RETIRED_COORD_AXES`] rather than past [`COORD_AXES`] — which is what
/// keeps them where they were when there were four axes. The gap between
/// the two is the reason `Var`'s three meanings do not collide.
pub(crate) const REDUCE_BINDER_BASE: u8 = COORD_AXES as u8 + RETIRED_COORD_AXES.len() as u8;

/// How many reduction binders the index space holds, starting at
/// [`REDUCE_BINDER_BASE`] — the depth of nested folds a kernel may carry.
pub(crate) const REDUCE_BINDERS: u8 = 4;

// ───────────────────────────────────────── ExprId ─────────────────────────────

/// Index into an [`ExprArena`]. Copy, 4 bytes, no refcount.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ExprId(pub u32);

// ───────────────────────────────────────── Buffers ────────────────────────────

/// Slot index into an [`ExprArena`]'s buffer table. Copy, 2 bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct BufferId(pub u16);

/// Which block of memory a declaration refers to, independent of any arena.
///
/// [`BufferId`] is a slot index into *one* arena's table, so it cannot answer
/// "the same buffer?" across a merge — two fragments each call their own
/// buffer slot 0. Extents cannot answer it either: two atlases of equal size
/// are a coincidence, not a fact, and treating them as one would bind a single
/// pointer for both and silently read the wrong pixels.
///
/// So identity is provenance. You get one by minting it, and copy it into
/// every declaration that names that memory; nothing else can collide with it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct BufferIdentity(u32);

impl BufferIdentity {
    /// Mint an identity distinct from every other in this process.
    ///
    /// # Panics
    ///
    /// Panics if the counter is exhausted, rather than wrapping onto a live
    /// identity and aliasing two unrelated buffers.
    #[must_use]
    pub fn mint() -> Self {
        static NEXT: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
        Self(mint_identity(&NEXT, "BufferIdentity"))
    }
}

/// The one counter discipline behind every provenance identity.
///
/// `fetch_add` + assert was wrong: the add WRAPS before the assert fires, so
/// if that panic is ever caught — or merely unwinds a non-fatal worker thread
/// — the counter has already returned to 0 and the next mint hands out an
/// identity that is still live. Two unrelated buffers (or uniforms) would
/// then compare identical and merge into one splice/JIT slot. `fetch_update`
/// declining to store leaves the counter permanently exhausted instead.
fn mint_identity(counter: &core::sync::atomic::AtomicU32, what: &str) -> u32 {
    counter
        .fetch_update(
            core::sync::atomic::Ordering::Relaxed,
            core::sync::atomic::Ordering::Relaxed,
            |n| n.checked_add(1),
        )
        .unwrap_or_else(|_| panic!("{what}: counter exhausted"))
}

/// Declaration of a bound memory buffer: the static shape of a collapsed
/// lattice. The extents are part of the IR (like a `Const`) even though the
/// contents are bound later, at JIT-compile time. Static extents are what
/// allow the emitter to fold address arithmetic, drop provably in-bounds
/// clamps, and unroll reduction loops.
///
/// Layout is row-major with `stride == width`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BufferDecl {
    /// Which memory this names. Two declarations of the same buffer must carry
    /// the same identity — that is what lets a merge collapse them into one
    /// slot instead of binding the pointer twice.
    pub id: BufferIdentity,
    /// X extent (samples per row).
    pub width: u32,
    /// Y extent (number of rows).
    pub height: u32,
}

// ───────────────────────────────────────── Uniforms ───────────────────────────

/// Slot index into an [`ExprArena`]'s uniform table. Copy, 2 bytes. Not an
/// identity: two arenas each call their own first uniform slot 0.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct UniformId(pub u16);

/// Which uniform a declaration refers to, independent of any arena.
///
/// A uniform is a scalar that is invariant across the lattice and supplied
/// at call time — the JIT tier's spelling of a builder's struct field. Its
/// identity is the *instance*: two instances of the same builder are two
/// factors of the kernel's parameter space, and one instance read from
/// twenty places is one factor. Neither a name nor a fragment-local index can
/// say that across a splice, so, exactly as for [`BufferIdentity`], identity
/// is provenance: minted once, copied into every declaration of the instance.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct UniformIdentity(u32);

impl UniformIdentity {
    /// Mint an identity distinct from every other in this process.
    ///
    /// # Panics
    ///
    /// Panics if the counter is exhausted, rather than wrapping onto a live
    /// identity and aliasing two unrelated uniforms.
    #[must_use]
    pub fn mint() -> Self {
        static NEXT: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
        Self(mint_identity(&NEXT, "UniformIdentity"))
    }
}

/// Declaration of a uniform: its identity plus the value the kernel holds
/// for it when nothing has been bound. The default is part of the IR so that
/// a bake without a block, and the scalar oracle, are total.
///
/// Compared and hashed by the default's *bit pattern*, so a declaration is a
/// plain key: `-0.0` and `0.0` are two defaults, and a NaN default is equal
/// to itself.
#[derive(Clone, Copy, Debug)]
pub struct UniformDecl {
    /// Which instance this names.
    pub id: UniformIdentity,
    /// The value bound when no block supplies one.
    pub default: f32,
}

impl PartialEq for UniformDecl {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.default.to_bits() == other.default.to_bits()
    }
}

impl Eq for UniformDecl {}

impl core::hash::Hash for UniformDecl {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.default.to_bits().hash(state);
    }
}

// ───────────────────────────────────────── ExprNode ───────────────────────────

/// A single expression node stored in the arena.
///
/// Layout is kept tight: the static assertion below guarantees <= 16 bytes.
#[derive(Clone, Debug, PartialEq)]
pub enum ExprNode {
    /// A bound variable: a lattice coordinate ([`COORD_AXES`] of them, X and
    /// Y), a reduction binder's index (`4..8`), or — in the macro front end
    /// only, before substitution — a parameter placeholder. Which one an
    /// index means is [`ExprArena::push_var`]'s documentation; the indices
    /// between the coordinates and the binders are not a hole to grow into,
    /// they are where Z and W used to be and nothing may claim them.
    Var(u8),
    Const(f32),
    Param(u8),
    /// Bound-memory leaf: references a [`BufferDecl`] in the arena's buffer
    /// table. Read through `Ternary(OpKind::Gather, buffer, x, y)`.
    Buffer(BufferId),
    /// Lattice-invariant scalar supplied per call: references a
    /// [`UniformDecl`] in the arena's uniform table. Never folded — its value
    /// is unknown until the call — and constant across the lattice, so it is
    /// loaded once per call rather than once per batch.
    Uniform(UniformId),
    /// A kernel named by content — *evaluate that kernel here*. The one node
    /// composition can hold instead of splicing a body in
    /// (docs/plans/2026-09-09-composition-is-linking.md); the referent lives
    /// in the [`KernelStore`](crate::store::KernelStore) and
    /// [`expand_refs`](crate::passes::expand_refs) is what puts it back.
    ///
    /// A leaf with an identity of its own, like [`ExprNode::Buffer`]: it has
    /// no children in *this* arena, and every pass that reads structure must
    /// either expand it or refuse it — never walk through it.
    Ref(KernelKey),
    Unary(OpKind, ExprId),
    Binary(OpKind, ExprId, ExprId),
    Ternary(OpKind, ExprId, ExprId, ExprId),
    /// N-ary node. Children live in `ExprArena::nary_children[start..start+len]`.
    Nary(OpKind, u32, u16),
    /// A bounded fold: `⊕_{k ∈ fold.range()} body[fold.binder() := k]`.
    ///
    /// The only node that *binds* — the binder is not free in the result — and
    /// the only one whose metadata is part of its identity rather than a
    /// child. It used to be `Nary(Reduce, [Const(op), Const(var), Const(n),
    /// body])`, decoded by four readers with three different failure modes;
    /// see [`crate::fold`] for why an e-graph could not hold that shape.
    Reduce {
        fold: Fold,
        body: ExprId,
    },
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
    /// Buffer declarations, indexed by [`BufferId`]. The memory analogue of
    /// the symbol table: shapes are static IR, contents are bound at JIT time.
    buffers: Vec<BufferDecl>,
    /// Uniform declarations, indexed by [`UniformId`]: the scalar arguments
    /// of the kernel, each with its default. Values are bound per call.
    uniforms: Vec<UniformDecl>,
}

impl Default for ExprArena {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprArena {
    /// Create an empty arena.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            nary_children: Vec::new(),
            buffers: Vec::new(),
            uniforms: Vec::new(),
        }
    }

    /// Create an arena pre-allocated for `n` nodes.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(n),
            nary_children: Vec::new(),
            buffers: Vec::new(),
            uniforms: Vec::new(),
        }
    }

    /// Truncate to zero nodes without deallocating backing storage.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.nary_children.clear();
        self.buffers.clear();
        self.uniforms.clear();
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
    ///
    /// Only `0..COORD_AXES` are lattice coordinates. [`RETIRED_COORD_AXES`]
    /// are the Z and W axes, which no longer exist: an arena that names one
    /// is refused where it would become code
    /// ([`Kernel::from_parts`](crate::Kernel::from_parts), and the JIT cache),
    /// rather than here, because the same node is also a reduction binder's
    /// index and a rewrite rule's pattern metavariable, and those namespaces
    /// are dense from zero.
    pub fn push_var(&mut self, i: u8) -> ExprId {
        self.push_node(ExprNode::Var(i))
    }

    /// The retired coordinate axis reachable from `root`, if any — the guard
    /// [`Kernel::from_parts`](crate::Kernel::from_parts) and
    /// `emit::compile` apply before an arena can become a compiled kernel.
    ///
    /// A `Var(2)` reaching the emitter would read the third base coordinate,
    /// which a collapse passes as zero and no longer means anything: the
    /// pixels would be plausible and wrong. Refusing it is what makes "no
    /// emitted kernel reads the retired lanes" a fact rather than a habit.
    ///
    /// **Reachable from `root`, not every node.** An arena is a bump list and
    /// nothing prunes it: `substitute_vars_with` rewrites the reachable graph
    /// and leaves what it replaced behind, so an arena whose retired axes were
    /// correctly substituted still *holds* the original `Var(2)` nodes. They
    /// are not emitted, because nothing reaches them. Scanning every node
    /// refuses that arena and is simply wrong — it cost this change a CI round
    /// trip.
    #[must_use]
    pub fn retired_axis(&self, root: ExprId) -> Option<u8> {
        let mut seen = alloc::vec![false; self.nodes.len()];
        let mut stack = alloc::vec![root];
        while let Some(id) = stack.pop() {
            let idx = id.0 as usize;
            if core::mem::replace(&mut seen[idx], true) {
                continue;
            }
            if let ExprNode::Var(i) = &self.nodes[idx]
                && RETIRED_COORD_AXES.contains(i)
            {
                return Some(*i);
            }
            stack.extend(self.children(id));
        }
        None
    }

    /// The first `Var(i)` with `i >= floor` reachable from `root`, if any.
    ///
    /// `Var`'s index space is three namespaces stacked in one integer —
    /// coordinates, then the reserved retired axes, then reduction binders,
    /// then a binder's under-construction placeholder — so "is this term open
    /// above `floor`?" is the only question a caller can ask structurally.
    /// [`retired_axis`](ExprArena::retired_axis) is its sibling for the one
    /// range that is closed rather than open-ended.
    ///
    /// Reachable from `root`, not every node, for
    /// [`retired_axis`](ExprArena::retired_axis)'s reason: an arena keeps the
    /// nodes a rebuild replaced, and nothing evaluates those.
    #[must_use]
    pub fn free_var_at_or_above(&self, root: ExprId, floor: u8) -> Option<u8> {
        let mut seen = alloc::vec![false; self.nodes.len()];
        let mut stack = alloc::vec![root];
        while let Some(id) = stack.pop() {
            let idx = id.0 as usize;
            if core::mem::replace(&mut seen[idx], true) {
                continue;
            }
            if let ExprNode::Var(i) = &self.nodes[idx]
                && *i >= floor
            {
                return Some(*i);
            }
            stack.extend(self.children(id));
        }
        None
    }

    /// Push a `Const(v)` node.
    pub fn push_const(&mut self, v: f32) -> ExprId {
        self.push_node(ExprNode::Const(v))
    }

    /// Push a `Param(i)` node.
    pub fn push_param(&mut self, i: u8) -> ExprId {
        self.push_node(ExprNode::Param(i))
    }

    /// Declare a buffer slot, returning its [`BufferId`].
    ///
    /// # Panics
    ///
    /// Panics if the buffer table is full (`u16::MAX` slots).
    pub fn declare_buffer(&mut self, decl: BufferDecl) -> BufferId {
        assert!(
            self.buffers.len() < u16::MAX as usize,
            "declare_buffer: buffer table full ({} slots)",
            self.buffers.len()
        );
        let id = BufferId(self.buffers.len() as u16);
        self.buffers.push(decl);
        id
    }

    /// Push a `Buffer(id)` leaf node.
    ///
    /// # Panics
    ///
    /// Panics if `id` has not been declared via [`ExprArena::declare_buffer`].
    pub fn push_buffer(&mut self, id: BufferId) -> ExprId {
        assert!(
            (id.0 as usize) < self.buffers.len(),
            "push_buffer: BufferId({}) not declared (table has {} entries)",
            id.0,
            self.buffers.len()
        );
        self.push_node(ExprNode::Buffer(id))
    }

    /// Declare a uniform slot, returning its [`UniformId`].
    ///
    /// # Panics
    ///
    /// Panics if the uniform table is full (`u16::MAX` slots).
    pub fn declare_uniform(&mut self, decl: UniformDecl) -> UniformId {
        assert!(
            self.uniforms.len() < u16::MAX as usize,
            "declare_uniform: uniform table full ({} slots)",
            self.uniforms.len()
        );
        let id = UniformId(self.uniforms.len() as u16);
        self.uniforms.push(decl);
        id
    }

    /// The slot naming `decl`'s instance in this arena, declaring one if this
    /// is the first time that identity has been seen.
    ///
    /// Two declarations of one identity with different defaults cannot happen
    /// through the handle that minted it — the default travels with the
    /// identity in one `Copy` value — so it is a corrupt graph, and an
    /// assertion rather than a silent alias onto whichever arrived first.
    pub(crate) fn uniform_slot_for(&mut self, decl: UniformDecl) -> UniformId {
        match self.uniforms.iter().position(|d| d.id == decl.id) {
            Some(i) => {
                assert_eq!(
                    self.uniforms[i], decl,
                    "two declarations share a UniformIdentity but disagree on the default"
                );
                UniformId(i as u16)
            }
            None => self.declare_uniform(decl),
        }
    }

    /// Push a `Uniform(id)` leaf node.
    ///
    /// # Panics
    ///
    /// Panics if `id` has not been declared via [`ExprArena::declare_uniform`].
    pub fn push_uniform(&mut self, id: UniformId) -> ExprId {
        assert!(
            (id.0 as usize) < self.uniforms.len(),
            "push_uniform: UniformId({}) not declared (table has {} entries)",
            id.0,
            self.uniforms.len()
        );
        self.push_node(ExprNode::Uniform(id))
    }

    /// Push a `Ref(key)` leaf — a kernel named by content rather than spliced
    /// in.
    ///
    /// No table declares it and nothing here checks that `key` resolves: the
    /// referent lives in the process-global
    /// [`KernelStore`](crate::store::KernelStore), which is the only thing
    /// that can answer, and [`expand_refs`](crate::passes::expand_refs) is
    /// where an unknown key is reported. `Kernel::by_ref` is the only
    /// producer.
    pub fn push_ref(&mut self, key: KernelKey) -> ExprId {
        self.push_node(ExprNode::Ref(key))
    }

    /// Get the declaration for a uniform slot.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of bounds.
    #[inline]
    #[must_use]
    pub fn uniform_decl(&self, id: UniformId) -> &UniformDecl {
        &self.uniforms[id.0 as usize]
    }

    /// All declared uniforms, indexed by [`UniformId`].
    #[inline]
    #[must_use]
    pub fn uniforms(&self) -> &[UniformDecl] {
        &self.uniforms
    }

    /// Push a `Gather(buffer, x, y)` read of a declared buffer.
    ///
    /// Semantics: floor the indices, clamp to the declared extents, gather
    /// row-major. `DiscreteManifold::kernel` is exactly one of these.
    pub fn push_gather(&mut self, buffer: BufferId, x: ExprId, y: ExprId) -> ExprId {
        let buf = self.push_buffer(buffer);
        self.push_ternary(OpKind::Gather, buf, x, y)
    }

    /// Push the bounded fold `⊕_{k ∈ fold.range()} body[fold.binder() := k]`.
    ///
    /// Two arguments, because [`Fold`] is the metadata: which algebra, which
    /// index, which range. Every one of those was an assertion here — a
    /// combiner that is a monoid, a var index inside the binder space, a trip
    /// count that fits — and each is now a thing the type will not build.
    /// `expand_reduce` lowers a survivor to an unrolled accumulation.
    pub fn push_reduce(&mut self, fold: Fold, body: ExprId) -> ExprId {
        self.push_node(ExprNode::Reduce { fold, body })
    }

    /// Get the declaration for a buffer slot.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of bounds.
    #[inline]
    #[must_use]
    pub fn buffer_decl(&self, id: BufferId) -> &BufferDecl {
        &self.buffers[id.0 as usize]
    }

    /// All declared buffers, indexed by [`BufferId`].
    #[inline]
    #[must_use]
    pub fn buffers(&self) -> &[BufferDecl] {
        &self.buffers
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

    // ───────────────────── node observation ───────

    /// Visit node payloads in construction order.
    ///
    /// The order is children before parents, but the iterator deliberately
    /// does not expose the backing allocation or the n-ary child slab. Code
    /// that needs an expression's edges must use [`Self::children`].
    #[inline]
    #[must_use]
    pub fn nodes(&self) -> impl ExactSizeIterator<Item = &ExprNode> + DoubleEndedIterator + '_ {
        self.nodes.iter()
    }

    /// Reconstruct an arena from raw parts.
    ///
    /// # Safety contract (logical, not `unsafe`)
    ///
    /// The caller must ensure that every `ExprId` referenced by nodes in
    /// `nodes` is in-bounds, and that `Nary` start/len pairs index validly
    /// into `nary_children`. Violating this will cause panics on access,
    /// not UB.
    /// The reconstructed arena has empty buffer and uniform tables, so it
    /// cannot hold `Buffer` or `Uniform` nodes.
    #[must_use]
    pub fn from_raw(nodes: Vec<ExprNode>, nary_children: Vec<ExprId>) -> Self {
        Self {
            nodes,
            nary_children,
            buffers: Vec::new(),
            uniforms: Vec::new(),
        }
    }

    // ───────────────────── access ────────────────────────────

    /// Get the node at `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of bounds.
    #[inline]
    #[must_use]
    pub fn node(&self, id: ExprId) -> &ExprNode {
        &self.nodes[id.0 as usize]
    }

    /// Get the N-ary children slice for a `Nary(_, start, len)` node.
    ///
    /// # Panics
    ///
    /// Panics if `start + len` exceeds the internal nary_children buffer.
    #[inline]
    #[must_use]
    pub fn nary_children_slice(&self, start: u32, len: u16) -> &[ExprId] {
        let s = start as usize;
        let l = len as usize;
        &self.nary_children[s..s + l]
    }

    /// Get the [`OpKind`] of the node at `id`.
    ///
    /// Leaf nodes map to: `Var -> OpKind::Var`, `Const/Param -> OpKind::Const`,
    /// `Buffer -> OpKind::Buffer`, `Uniform -> OpKind::Uniform`.
    ///
    /// # Panics
    ///
    /// Panics on an [`ExprNode::Ref`]. A reference is a *name*, not an
    /// operation: giving it an `OpKind` would let it into every cost model,
    /// vocabulary and emitter that dispatches on one, and each of those would
    /// then price or emit a kernel it cannot see. Expand it first.
    #[inline]
    #[must_use]
    pub fn kind(&self, id: ExprId) -> OpKind {
        match &self.nodes[id.0 as usize] {
            ExprNode::Var(_) => OpKind::Var,
            ExprNode::Const(_) | ExprNode::Param(_) => OpKind::Const,
            ExprNode::Buffer(_) => OpKind::Buffer,
            ExprNode::Uniform(_) => OpKind::Uniform,
            ExprNode::Ref(key) => panic!(
                "ExprArena::kind: {key:?} is a reference to a kernel, not an \
                 operation; run passes::expand_refs before asking for a kind"
            ),
            ExprNode::Unary(op, _) => *op,
            ExprNode::Binary(op, _, _) => *op,
            ExprNode::Ternary(op, _, _, _) => *op,
            ExprNode::Nary(op, _, _) => *op,
            ExprNode::Reduce { .. } => OpKind::Reduce,
        }
    }

    /// Iterate over the child [`ExprId`]s of the node at `id`.
    #[inline]
    #[must_use]
    pub fn children(&self, id: ExprId) -> ExprChildren<'_> {
        match &self.nodes[id.0 as usize] {
            ExprNode::Var(_)
            | ExprNode::Const(_)
            | ExprNode::Param(_)
            | ExprNode::Buffer(_)
            | ExprNode::Uniform(_)
            | ExprNode::Ref(_) => ExprChildren::Zero,
            ExprNode::Unary(_, a) => ExprChildren::One(*a),
            ExprNode::Binary(_, a, b) => ExprChildren::Two(*a, *b),
            ExprNode::Ternary(_, a, b, c) => ExprChildren::Three(*a, *b, *c),
            ExprNode::Nary(_, start, len) => {
                let s = *start as usize;
                let l = *len as usize;
                ExprChildren::Nary(&self.nary_children[s..s + l])
            }
            // One child, not four: the combiner, the binder and the extent
            // are no longer expressions, so nothing that walks children can
            // reach them, fold them, or cost them.
            ExprNode::Reduce { body, .. } => ExprChildren::One(*body),
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
                ExprNode::Var(_)
                | ExprNode::Const(_)
                | ExprNode::Param(_)
                | ExprNode::Buffer(_)
                | ExprNode::Uniform(_)
                | ExprNode::Ref(_) => {
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
                ExprNode::Reduce { body, .. } => stack.push((*body, d + 1)),
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
                ExprNode::Const(_)
                | ExprNode::Param(_)
                | ExprNode::Buffer(_)
                | ExprNode::Uniform(_)
                | ExprNode::Ref(_) => {}
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
                ExprNode::Reduce { body, .. } => stack.push(*body),
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
                ExprNode::Var(_)
                | ExprNode::Const(_)
                | ExprNode::Param(_)
                | ExprNode::Buffer(_)
                | ExprNode::Uniform(_)
                | ExprNode::Ref(_) => {}
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
                ExprNode::Reduce { body, .. } => stack.push(*body),
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
                ExprNode::Var(_)
                | ExprNode::Const(_)
                | ExprNode::Param(_)
                | ExprNode::Buffer(_)
                | ExprNode::Uniform(_)
                | ExprNode::Ref(_) => {}
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
                ExprNode::Reduce { body, .. } => stack.push(*body),
            }
        }
        count
    }

    /// Replace every `Param(i)` node with what `params[i]` says it is: a
    /// `Const` folded into the fragment, or a `Uniform` slot declared for the
    /// handle's identity (one slot per identity, however many placeholders
    /// name it).
    ///
    /// Returns the new root [`ExprId`] in the **same** arena. Old nodes become
    /// unreachable garbage — that is fine for an append-only arena.
    ///
    /// # Panics
    ///
    /// Panics if any `Param(i)` has `i >= params.len()`.
    pub fn substitute_params(&mut self, root: ExprId, params: &[Scalar]) -> ExprId {
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
                        ExprNode::Var(_)
                        | ExprNode::Const(_)
                        | ExprNode::Param(_)
                        | ExprNode::Buffer(_)
                        | ExprNode::Uniform(_)
                        | ExprNode::Ref(_) => {}
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
                        ExprNode::Reduce { body, .. } => work.push(Task::Descend(*body)),
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
                            match params[idx] {
                                Scalar::Const(v) => self.push_const(v),
                                Scalar::Uniform(u) => {
                                    let slot = self.uniform_slot_for(u.decl());
                                    self.push_uniform(slot)
                                }
                            }
                        }
                        ExprNode::Var(i) => self.push_var(i),
                        ExprNode::Const(v) => self.push_const(v),
                        // Buffer and uniform ids stay valid: the tables live
                        // in this arena.
                        ExprNode::Buffer(b) => self.push_node(ExprNode::Buffer(b)),
                        ExprNode::Uniform(u) => self.push_node(ExprNode::Uniform(u)),
                        // A key is arena-independent, so a reference copies
                        // across as itself.
                        ExprNode::Ref(k) => self.push_ref(k),
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
                        ExprNode::Reduce { fold, body } => {
                            let body = id_map[body.0 as usize]
                                .expect("substitute_params: reduce body not yet mapped");
                            self.push_reduce(fold, body)
                        }
                    };
                    id_map[id.0 as usize] = Some(new_id);
                }
            }
        }

        id_map[root.0 as usize].expect("substitute_params: root was never mapped")
    }

    // ───────────────────── composition (P4: arena splicing) ─────────────────

    /// Copy the fragment reachable from `root` in `other` into this arena,
    /// returning the fragment's new root here. Shared subexpressions are
    /// copied once, so a DAG stays a DAG. This is the substrate of kernel
    /// composition: the spliced fragment reads this arena's coordinate
    /// variables directly (an identity contramap); warp it afterwards with
    /// [`ExprArena::substitute_vars_with`].
    ///
    /// Buffers the fragment reads are merged into this arena's table by
    /// [`BufferIdentity`] and its `Buffer` leaves remapped, so a sampler
    /// composes like anything else and reading the same memory from twenty
    /// places still binds one pointer. Uniforms merge the same way, by
    /// [`UniformIdentity`]: one instance read from twenty places is one slot,
    /// and two instances of one builder stay two.
    pub fn splice(&mut self, other: &ExprArena, root: ExprId) -> ExprId {
        let mut id_map: Vec<Option<ExprId>> = vec![None; other.nodes.len()];
        // Fragment-local BufferId -> this arena's slot, filled lazily.
        let mut buf_map: Vec<Option<BufferId>> = vec![None; other.buffers.len()];
        let mut uni_map: Vec<Option<UniformId>> = vec![None; other.uniforms.len()];

        enum Task {
            Descend(ExprId),
            Emit(ExprId),
        }
        let mut work: Vec<Task> = vec![Task::Descend(root)];

        while let Some(task) = work.pop() {
            match task {
                Task::Descend(id) => {
                    if id_map[id.0 as usize].is_some() {
                        continue;
                    }
                    work.push(Task::Emit(id));
                    let children: Vec<ExprId> = other.children(id).collect();
                    for child in children.into_iter().rev() {
                        work.push(Task::Descend(child));
                    }
                }
                Task::Emit(id) => {
                    if id_map[id.0 as usize].is_some() {
                        continue;
                    }
                    let m = |old: ExprId| {
                        id_map[old.0 as usize].expect("splice: child copied before parent")
                    };
                    let new_id = match other.nodes[id.0 as usize].clone() {
                        ExprNode::Var(i) => self.push_var(i),
                        ExprNode::Const(v) => self.push_const(v),
                        ExprNode::Param(i) => self.push_param(i),
                        // Content-addressed, so a reference means the same
                        // kernel in every arena and needs no remapping.
                        ExprNode::Ref(k) => self.push_ref(k),
                        ExprNode::Buffer(b) => {
                            let slot = match buf_map[b.0 as usize] {
                                Some(slot) => slot,
                                None => {
                                    let decl = other.buffers[b.0 as usize];
                                    let slot =
                                        match self.buffers.iter().position(|d| d.id == decl.id) {
                                            Some(i) => {
                                                assert_eq!(
                                                    self.buffers[i], decl,
                                                    "splice: two declarations share a \
                                                 BufferIdentity but disagree on extents"
                                                );
                                                BufferId(i as u16)
                                            }
                                            None => self.declare_buffer(decl),
                                        };
                                    buf_map[b.0 as usize] = Some(slot);
                                    slot
                                }
                            };
                            self.push_buffer(slot)
                        }
                        ExprNode::Uniform(u) => {
                            let slot = match uni_map[u.0 as usize] {
                                Some(slot) => slot,
                                None => {
                                    let slot = self.uniform_slot_for(other.uniforms[u.0 as usize]);
                                    uni_map[u.0 as usize] = Some(slot);
                                    slot
                                }
                            };
                            self.push_uniform(slot)
                        }
                        ExprNode::Unary(op, a) => {
                            let a = m(a);
                            self.push_unary(op, a)
                        }
                        ExprNode::Binary(op, a, b) => {
                            let (a, b) = (m(a), m(b));
                            self.push_binary(op, a, b)
                        }
                        ExprNode::Ternary(op, a, b, c) => {
                            let (a, b, c) = (m(a), m(b), m(c));
                            self.push_ternary(op, a, b, c)
                        }
                        ExprNode::Nary(op, start, len) => {
                            let (s, l) = (start as usize, len as usize);
                            let mapped: Vec<ExprId> = other.nary_children[s..s + l]
                                .iter()
                                .map(|c| m(*c))
                                .collect();
                            self.push_nary(op, &mapped)
                        }
                        ExprNode::Reduce { fold, body } => {
                            let body = m(body);
                            self.push_reduce(fold, body)
                        }
                    };
                    id_map[id.0 as usize] = Some(new_id);
                }
            }
        }

        id_map[root.0 as usize].expect("splice: root was never copied")
    }

    /// Rebuild the subgraph at `root`, replacing every `Var(i)` for which
    /// `subs` has an entry with the given (already existing) node — the
    /// generic contramap: a coordinate warp substitutes `Var(0..4)` with
    /// coordinate expressions, which is what `Kernel::at` is built from.
    ///
    /// Entries must reference nodes already in this arena (e.g. from
    /// [`ExprArena::splice`]). Unlisted variables are preserved. Returns the
    /// new root in the same arena; old nodes become unreachable garbage, as
    /// with [`ExprArena::substitute_params`].
    pub fn substitute_vars_with(&mut self, root: ExprId, subs: &[(u8, ExprId)]) -> ExprId {
        let lookup = |i: u8| subs.iter().find(|(v, _)| *v == i).map(|(_, id)| *id);

        let old_len = self.nodes.len();
        let mut id_map: Vec<Option<ExprId>> = vec![None; old_len];

        enum Task {
            Descend(ExprId),
            Emit(ExprId),
        }
        let mut work: Vec<Task> = vec![Task::Descend(root)];

        while let Some(task) = work.pop() {
            match task {
                Task::Descend(id) => {
                    if id_map[id.0 as usize].is_some() {
                        continue;
                    }
                    work.push(Task::Emit(id));
                    let children: Vec<ExprId> = self.children(id).collect();
                    for child in children.into_iter().rev() {
                        work.push(Task::Descend(child));
                    }
                }
                Task::Emit(id) => {
                    if id_map[id.0 as usize].is_some() {
                        continue;
                    }
                    let m = |old: ExprId| {
                        id_map[old.0 as usize]
                            .expect("substitute_vars_with: child rebuilt before parent")
                    };
                    let new_id = match self.nodes[id.0 as usize].clone() {
                        ExprNode::Var(i) => match lookup(i) {
                            Some(replacement) => replacement,
                            None => self.push_var(i),
                        },
                        ExprNode::Const(v) => self.push_const(v),
                        ExprNode::Param(i) => self.push_param(i),
                        ExprNode::Buffer(b) => self.push_node(ExprNode::Buffer(b)),
                        ExprNode::Uniform(u) => self.push_node(ExprNode::Uniform(u)),
                        ExprNode::Ref(k) => self.push_ref(k),
                        ExprNode::Unary(op, a) => {
                            let a = m(a);
                            self.push_unary(op, a)
                        }
                        ExprNode::Binary(op, a, b) => {
                            let (a, b) = (m(a), m(b));
                            self.push_binary(op, a, b)
                        }
                        ExprNode::Ternary(op, a, b, c) => {
                            let (a, b, c) = (m(a), m(b), m(c));
                            self.push_ternary(op, a, b, c)
                        }
                        ExprNode::Nary(op, start, len) => {
                            let (s, l) = (start as usize, len as usize);
                            let child_ids: Vec<ExprId> = self.nary_children[s..s + l].to_vec();
                            let mapped: Vec<ExprId> = child_ids.into_iter().map(m).collect();
                            self.push_nary(op, &mapped)
                        }
                        ExprNode::Reduce { fold, body } => {
                            let body = m(body);
                            self.push_reduce(fold, body)
                        }
                    };
                    id_map[id.0 as usize] = Some(new_id);
                }
            }
        }

        id_map[root.0 as usize].expect("substitute_vars_with: root was never rebuilt")
    }

    // ───────────────────── linking ───────────────────────────

    /// The subgraph reachable from `root`, with its buffer and uniform tables
    /// replaced by the given orders — the link step: slot `i` of the result
    /// names `buffers[i]` / `uniforms[i]` and every reachable leaf is
    /// remapped to its slot there. Reachable nodes keep their relative order
    /// (ascending id, so still topological), which is the order a schedule
    /// is built in; construction garbage is dropped, which no schedule ever
    /// saw. So nothing downstream of the tables — the schedule, the
    /// registers, the bytes — can move.
    ///
    /// A declaration this arena holds but no reachable node reads is left
    /// behind with the garbage: `Kernel::at` splices all four coordinate
    /// fragments whether or not the receiver reads that axis, so a table
    /// routinely names an instance the graph does not, and the link — which
    /// is computed over the reachable subgraph — rightly omits it. The orders
    /// may likewise declare identities nothing here reads; those slots exist
    /// in the result unread.
    ///
    /// # Panics
    ///
    /// Panics if a *reachable* leaf's declaration has no entry in the given
    /// order, or disagrees with it (extents, default).
    #[must_use]
    pub fn relink(
        &self,
        root: ExprId,
        buffers: &[BufferDecl],
        uniforms: &[UniformDecl],
    ) -> (ExprArena, ExprId) {
        let mut reachable = vec![false; self.nodes.len()];
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut reachable[id.0 as usize], true) {
                continue;
            }
            stack.extend(self.children(id));
        }

        let buffer_slot = |b: BufferId| -> BufferId {
            let decl = self.buffers[b.0 as usize];
            let i = buffers
                .iter()
                .position(|d| d.id == decl.id)
                .unwrap_or_else(|| panic!("relink: reachable {decl:?} is not in the link"));
            assert_eq!(buffers[i], decl, "relink: buffer declaration disagrees");
            BufferId(i as u16)
        };
        let uniform_slot = |u: UniformId| -> UniformId {
            let decl = self.uniforms[u.0 as usize];
            let i = uniforms
                .iter()
                .position(|d| d.id == decl.id)
                .unwrap_or_else(|| panic!("relink: reachable {decl:?} is not in the link"));
            assert_eq!(uniforms[i], decl, "relink: uniform declaration disagrees");
            UniformId(i as u16)
        };

        let mut out = ExprArena {
            nodes: Vec::with_capacity(self.nodes.len()),
            nary_children: Vec::new(),
            buffers: buffers.to_vec(),
            uniforms: uniforms.to_vec(),
        };
        let mut dense: Vec<Option<ExprId>> = vec![None; self.nodes.len()];
        for (idx, node) in self.nodes.iter().enumerate() {
            if !reachable[idx] {
                continue;
            }
            let m =
                |old: ExprId| dense[old.0 as usize].expect("relink: child densified before parent");
            let new_id = match node {
                ExprNode::Var(i) => out.push_var(*i),
                ExprNode::Const(v) => out.push_const(*v),
                ExprNode::Param(i) => out.push_param(*i),
                ExprNode::Buffer(b) => out.push_buffer(buffer_slot(*b)),
                ExprNode::Uniform(u) => out.push_uniform(uniform_slot(*u)),
                ExprNode::Ref(k) => out.push_ref(*k),
                ExprNode::Unary(op, a) => out.push_unary(*op, m(*a)),
                ExprNode::Binary(op, a, b) => out.push_binary(*op, m(*a), m(*b)),
                ExprNode::Ternary(op, a, b, c) => out.push_ternary(*op, m(*a), m(*b), m(*c)),
                ExprNode::Nary(op, start, len) => {
                    let (s, l) = (*start as usize, *len as usize);
                    let mapped: Vec<ExprId> =
                        self.nary_children[s..s + l].iter().map(|c| m(*c)).collect();
                    out.push_nary(*op, &mapped)
                }
                ExprNode::Reduce { fold, body } => {
                    let body = m(*body);
                    out.push_reduce(*fold, body)
                }
            };
            dense[idx] = Some(new_id);
        }
        let new_root = dense[root.0 as usize].expect("relink: root is reachable from itself");
        (out, new_root)
    }

    // ───────────────────── display ───────────────────────────

    /// Format the subtree rooted at `root` as an S-expression, matching the
    /// [`Expr`] display format.
    pub(crate) fn fmt_expr(&self, root: ExprId, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
                    ExprNode::Buffer(b) => write!(f, "Buffer({})", b.0)?,
                    ExprNode::Uniform(u) => write!(f, "Uniform({})", u.0)?,
                    ExprNode::Ref(k) => write!(f, "Ref({:#018x})", k.bits())?,
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
                    ExprNode::Reduce { fold, body } => {
                        stack.push(Task::WriteStr(")"));
                        stack.push(Task::Visit(*body));
                        write!(
                            f,
                            "{}_{}over({}..{})(",
                            OpKind::Reduce.name(),
                            fold.binder().var(),
                            fold.range().start,
                            fold.range().end
                        )?;
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
                // Buffer slots compare by id AND declared shape, so the
                // comparison is meaningful across arenas with different tables.
                (ExprNode::Buffer(sb), ExprNode::Buffer(ob)) => {
                    if sb != ob || self.buffers[sb.0 as usize] != other.buffers[ob.0 as usize] {
                        return false;
                    }
                }
                // Uniform slots likewise: by slot AND declaration.
                (ExprNode::Uniform(su), ExprNode::Uniform(ou)) => {
                    if su != ou || self.uniforms[su.0 as usize] != other.uniforms[ou.0 as usize] {
                        return false;
                    }
                }
                // A key IS the content, so comparing keys compares the
                // kernels named — no arena is needed to say so.
                (ExprNode::Ref(sk), ExprNode::Ref(ok)) => {
                    if sk != ok {
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
                (
                    ExprNode::Reduce {
                        fold: s_fold,
                        body: s_body,
                    },
                    ExprNode::Reduce {
                        fold: o_fold,
                        body: o_body,
                    },
                ) => {
                    if s_fold != o_fold {
                        return false;
                    }
                    stack.push((*s_body, *o_body));
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
    use crate::fold::{Binder, Monoid};
    use alloc::format;

    // 1. test_push_and_access
    #[test]
    fn push_and_access() {
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
    fn node_count() {
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
    fn verify_depth() {
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
    fn verify_has_var() {
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
    fn verify_has_degenerate() {
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
    fn clear_preserves_capacity() {
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
    fn verify_substitute_params() {
        let mut arena = ExprArena::new();
        let p0 = arena.push_param(0);
        let p1 = arena.push_param(1);
        let root = arena.push_binary(OpKind::Add, p0, p1);

        let new_root = arena.substitute_params(root, &[Scalar::Const(10.0), Scalar::Const(20.0)]);

        match arena.node(new_root) {
            ExprNode::Binary(OpKind::Add, a, b) => {
                assert!(matches!(arena.node(*a), ExprNode::Const(v) if (*v - 10.0).abs() < 1e-6));
                assert!(matches!(arena.node(*b), ExprNode::Const(v) if (*v - 20.0).abs() < 1e-6));
            }
            other => panic!("expected Binary(Add, ...), got {:?}", other),
        }
    }

    #[test]
    fn push_gather_should_create_node_when_valid() {
        let mut arena = ExprArena::new();
        let buf = arena.declare_buffer(BufferDecl {
            id: crate::arena::BufferIdentity::mint(),
            width: 16,
            height: 8,
        });
        assert_eq!(arena.buffers().len(), 1);
        assert_eq!(arena.buffer_decl(buf).width, 16);

        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let gather = arena.push_gather(buf, x, y);

        // Gather is a ternary whose first child is the Buffer leaf.
        assert_eq!(arena.kind(gather), OpKind::Gather);
        let children: Vec<ExprId> = arena.children(gather).collect();
        assert_eq!(children.len(), 3);
        assert!(matches!(arena.node(children[0]), ExprNode::Buffer(b) if *b == buf));
        assert_eq!(arena.kind(children[0]), OpKind::Buffer);
        assert_eq!(arena.children(children[0]).count(), 0); // Buffer is a leaf

        assert_eq!(format!("{}", arena.display(children[0])), "Buffer(0)");
    }

    #[test]
    #[should_panic(expected = "not declared")]
    fn push_buffer_should_panic_when_undeclared() {
        let mut arena = ExprArena::new();
        let _ = arena.push_buffer(BufferId(0));
    }

    /// A fold's metadata is a [`Fold`], so the two things `push_reduce` used
    /// to assert — a combiner that is a monoid, a var index inside the binder
    /// space — are no longer states this function can be *called* in. The
    /// tests that pinned those panics could not be written any more, and the
    /// properties they guarded are in `crate::fold`'s own tests instead. This
    /// is the whole point of the retype: an assertion you delete because the
    /// argument type refuses the value is the one kind you never have to
    /// maintain.
    #[test]
    fn a_fold_carries_its_own_metadata() {
        let mut arena = ExprArena::new();
        let body = arena.push_var(REDUCE_BINDER_BASE);
        let binder = Binder::from_var(REDUCE_BINDER_BASE).expect("the first binder");
        let fold = Fold::new(Monoid::SUM, binder, 0..4);
        let red = arena.push_reduce(fold, body);

        assert_eq!(arena.kind(red), OpKind::Reduce);
        // One child — the body. The combiner, the binder and the extent are
        // not expressions, so nothing that walks children can reach them.
        let children: Vec<ExprId> = arena.children(red).collect();
        assert_eq!(children, alloc::vec![body]);
        assert!(matches!(arena.node(red), ExprNode::Reduce { fold: f, .. } if *f == fold));
    }

    // 8. test_nary
    #[test]
    fn nary() {
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
    fn verify_display() {
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
    fn size_of_expr_node() {
        // Compile-time assertion exists above, but also verify at runtime.
        assert!(
            core::mem::size_of::<ExprNode>() <= 16,
            "ExprNode is {} bytes, expected <= 16",
            core::mem::size_of::<ExprNode>()
        );
    }

    #[test]
    fn expr_children_exact_size() {
        let mut arena = ExprArena::new();
        let v = arena.push_var(0);
        let c = arena.push_const(1.0);
        let bin = arena.push_binary(OpKind::Add, v, c);

        assert_eq!(arena.children(v).len(), 0);
        assert_eq!(arena.children(bin).len(), 2);
    }
}

#[cfg(test)]
mod composition_tests {
    use super::*;
    use crate::kind::OpKind;

    // ───────────────────────── uniforms ─────────────────────────

    /// A fragment reading one uniform, as `Uniform::kernel` would build it.
    fn uniform_fragment(decl: UniformDecl) -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let slot = a.declare_uniform(decl);
        let root = a.push_uniform(slot);
        (a, root)
    }

    fn uniform_decl(default: f32) -> UniformDecl {
        UniformDecl {
            id: UniformIdentity::mint(),
            default,
        }
    }

    #[test]
    #[should_panic(expected = "disagree on the default")]
    fn one_identity_with_two_defaults_is_refused() {
        let id = UniformIdentity::mint();
        let (donor, r) = uniform_fragment(UniformDecl { id, default: 1.0 });
        let mut host = ExprArena::new();
        let _ = host.declare_uniform(UniformDecl { id, default: 2.0 });
        let _ = host.splice(&donor, r);
    }

    #[test]
    fn f32_arguments_substitute_to_the_same_arena_as_before() {
        // The fold path is byte-for-byte what it was: `f32` keeps its meaning.
        let build = || {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let p = a.push_param(0);
            let root = a.push_binary(OpKind::Mul, x, p);
            (a, root)
        };
        let (mut folded, root) = build();
        let folded_root = folded.substitute_params(root, &[Scalar::from(2.5)]);
        let (mut by_hand, root) = build();
        let x = by_hand.push_var(0);
        let c = by_hand.push_const(2.5);
        let hand_root = by_hand.push_binary(OpKind::Mul, x, c);
        let _ = root;
        assert_eq!(
            folded.nodes().collect::<Vec<_>>(),
            by_hand.nodes().collect::<Vec<_>>()
        );
        assert_eq!(folded_root, hand_root);
        assert!(folded.uniforms().is_empty());
    }

    #[test]
    fn subtree_eq_distinguishes_uniform_declarations() {
        let (a, ra) = uniform_fragment(uniform_decl(1.0));
        let (b, rb) = uniform_fragment(uniform_decl(1.0));
        assert!(a.subtree_eq(ra, &a, ra));
        assert!(!a.subtree_eq(ra, &b, rb), "same slot, different instance");
    }
}
