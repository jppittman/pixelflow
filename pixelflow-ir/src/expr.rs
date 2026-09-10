//! The expression language: what a node *is*, and how a graph of them is
//! built, read, transformed and marshalled.
//!
//! Three things, kept apart on purpose:
//!
//! - **Topology** is [`Dag<ExprData>`](crate::dag::Dag): acyclicity, ordering,
//!   traversal, sharing. It knows nothing about expressions.
//! - **Payload** is [`ExprData`]: `Var`, `Const`, `Param`, `Buffer`,
//!   `Uniform`, `Op`. No child references, no edge indices, no slab offsets —
//!   arity is the DAG's `child_count`, not a variant.
//! - **Declarations** are an [`Environment`]: the buffer and uniform tables a
//!   `Buffer`/`Uniform` leaf indexes. [`Term`] is the pair, which is what
//!   every consumer of the IR actually needs.
//!
//! Construction goes through [`ExprBuilder`], which hands back an opaque
//! [`ExprRef`] and, at [`finish`](ExprBuilder::finish), a
//! [`Rooted<ExprData>`](crate::dag::Rooted) plus the environment its leaves
//! index. Reading is by [`Node`] handle. There is no index-addressed store in
//! between and nothing publishes one.

use alloc::vec::Vec;

use crate::dag::{Builder, Dag, Id, Node, Rooted, SideTable};
use crate::decl::{
    BufferDecl, BufferId, RETIRED_COORD_AXES, UniformDecl, UniformId, reduce_binders,
};
use crate::kernel::Scalar;
use crate::kind::{OpCode, OpKind};

/// Pure expression node payload.
///
/// Contains only operator or terminal identity; carries no child references,
/// edge indices, or slab offsets. All edge topology is owned by [`Dag<ExprData>`].
///
/// Constants are stored as IEEE 754 32-bit patterns (`u32`) for bit-exact
/// comparison and to implement `dag::Key` — bitwise `Eq`/`Ord`/`Hash`,
/// needed for interning — which an `f32`'s `NaN` would refuse.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum ExprData {
    /// Bound variable: a lattice coordinate (`0` for X, `1` for Y), a
    /// reduction binder's index (`4..8`), or a rewrite rule's metavariable.
    /// The indices between the coordinates and the binders are not a hole to
    /// grow into: they are where Z and W used to be
    /// ([`RETIRED_COORD_AXES`]) and nothing
    /// may claim them.
    Var(u8),
    /// 32-bit floating point literal stored as raw IEEE 754 bits.
    Const(u32),
    /// Macro parameter placeholder (0..). Valid only before kernel compilation.
    Param(u8),
    /// Bound-memory leaf: indexes an [`Environment`]'s buffer table. Read
    /// through `Op(Gather)` over `(buffer, x, y)`.
    Buffer(BufferId),
    /// Lattice-invariant scalar supplied per call: indexes an
    /// [`Environment`]'s uniform table. Never folded — its value is unknown
    /// until the call — and constant across the lattice, so it is loaded once
    /// per call rather than once per batch.
    Uniform(UniformId),
    /// Operator node. Arity (unary, binary, ternary, nary) is a property of the
    /// DAG edge count (`node.child_count()`).
    Op(OpKind),
}

impl ExprData {
    /// Create a constant node from an `f32`.
    #[inline]
    #[must_use]
    pub const fn constant(v: f32) -> Self {
        Self::Const(v.to_bits())
    }

    /// Read the constant value if this is a `Const` node.
    #[inline]
    #[must_use]
    pub const fn as_f32(self) -> Option<f32> {
        match self {
            Self::Const(b) => Some(f32::from_bits(b)),
            _ => None,
        }
    }

    /// Read the operation if this is an `Op` node.
    #[inline]
    #[must_use]
    pub const fn op(self) -> Option<OpKind> {
        match self {
            Self::Op(op) => Some(op),
            _ => None,
        }
    }

    /// Read the variable index if this is a `Var` node.
    #[inline]
    #[must_use]
    pub const fn var(self) -> Option<u8> {
        match self {
            Self::Var(v) => Some(v),
            _ => None,
        }
    }
}

/// Construction helpers on [`Builder<ExprData>`].
///
/// Crate-private: `Builder` is this crate's own construction machinery.
/// Outside callers build through [`ExprBuilder`].
pub(crate) trait ExprBuilderExt {
    fn push_var(&mut self, var: u8) -> Id;
    fn push_const(&mut self, val: f32) -> Id;
    fn push_param(&mut self, param: u8) -> Id;
    fn push_buffer(&mut self, buf: BufferId) -> Id;
    fn push_uniform(&mut self, uni: UniformId) -> Id;
    fn push_unary(&mut self, op: OpKind, child: Id) -> Id;
    fn push_binary(&mut self, op: OpKind, a: Id, b: Id) -> Id;
    fn push_ternary(&mut self, op: OpKind, a: Id, b: Id, c: Id) -> Id;
    fn push_nary(&mut self, op: OpKind, children: &[Id]) -> Id;
}

impl ExprBuilderExt for Builder<ExprData> {
    #[inline]
    fn push_var(&mut self, var: u8) -> Id {
        self.push_unique(ExprData::Var(var), &[])
    }

    #[inline]
    fn push_const(&mut self, val: f32) -> Id {
        self.push_unique(ExprData::constant(val), &[])
    }

    #[inline]
    fn push_param(&mut self, param: u8) -> Id {
        self.push_unique(ExprData::Param(param), &[])
    }

    #[inline]
    fn push_buffer(&mut self, buf: BufferId) -> Id {
        self.push_unique(ExprData::Buffer(buf), &[])
    }

    #[inline]
    fn push_uniform(&mut self, uni: UniformId) -> Id {
        self.push_unique(ExprData::Uniform(uni), &[])
    }

    #[inline]
    fn push_unary(&mut self, op: OpKind, child: Id) -> Id {
        self.push_unique(ExprData::Op(op), &[child])
    }

    #[inline]
    fn push_binary(&mut self, op: OpKind, a: Id, b: Id) -> Id {
        self.push_unique(ExprData::Op(op), &[a, b])
    }

    #[inline]
    fn push_ternary(&mut self, op: OpKind, a: Id, b: Id, c: Id) -> Id {
        self.push_unique(ExprData::Op(op), &[a, b, c])
    }

    #[inline]
    fn push_nary(&mut self, op: OpKind, children: &[Id]) -> Id {
        self.push_unique(ExprData::Op(op), children)
    }
}

// ────────────────────────────────────────── Environment & Term ────────────────

/// The declaration tables an expression's leaves index.
#[derive(Clone, Default, Debug, PartialEq)]
pub struct Environment {
    /// Buffer declarations, indexed by [`BufferId`]. The memory analogue of
    /// the symbol table: shapes are static IR, contents are bound at JIT time.
    pub buffers: Vec<BufferDecl>,
    /// Uniform declarations, indexed by [`UniformId`]: the scalar arguments
    /// of the kernel, each with its default. Values are bound per call.
    pub uniforms: Vec<UniformDecl>,
}

impl Environment {
    /// An environment declaring nothing.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The slot naming `decl`'s memory here, declaring one if this is the
    /// first time that identity has been seen.
    ///
    /// # Panics
    ///
    /// Panics if that identity is already declared with different extents —
    /// one memory described two ways is a corrupt graph, not a slot to alias
    /// onto whichever declaration arrived first.
    pub fn slot_for_buffer(&mut self, decl: BufferDecl) -> BufferId {
        match self.buffers.iter().position(|d| d.id == decl.id) {
            Some(i) => {
                assert_eq!(
                    self.buffers[i], decl,
                    "two declarations share a BufferIdentity but disagree on extents"
                );
                BufferId(i as u16)
            }
            None => {
                assert!(
                    self.buffers.len() < u16::MAX as usize,
                    "buffer table full ({} slots)",
                    self.buffers.len()
                );
                let id = BufferId(self.buffers.len() as u16);
                self.buffers.push(decl);
                id
            }
        }
    }

    /// The slot naming `decl`'s instance here, declaring one if this is the
    /// first time that identity has been seen.
    ///
    /// # Panics
    ///
    /// Panics if that identity is already declared with a different default.
    /// It cannot happen through the handle that minted it — the default
    /// travels with the identity in one `Copy` value — so it is a corrupt
    /// graph rather than a silent alias.
    pub fn slot_for_uniform(&mut self, decl: UniformDecl) -> UniformId {
        match self.uniforms.iter().position(|d| d.id == decl.id) {
            Some(i) => {
                assert_eq!(
                    self.uniforms[i], decl,
                    "two declarations share a UniformIdentity but disagree on the default"
                );
                UniformId(i as u16)
            }
            None => {
                assert!(
                    self.uniforms.len() < u16::MAX as usize,
                    "uniform table full ({} slots)",
                    self.uniforms.len()
                );
                let id = UniformId(self.uniforms.len() as u16);
                self.uniforms.push(decl);
                id
            }
        }
    }

    /// The declaration for a buffer slot.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of bounds.
    #[must_use]
    pub fn buffer(&self, id: BufferId) -> BufferDecl {
        self.buffers[id.0 as usize]
    }

    /// The declaration for a uniform slot.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of bounds.
    #[must_use]
    pub fn uniform(&self, id: UniformId) -> UniformDecl {
        self.uniforms[id.0 as usize]
    }
}

/// An expression node together with the tables its leaves index.
///
/// A `Buffer(3)` means nothing on its own — the slot is an index into *one*
/// environment — so every consumer that reads leaves needs the pair, and
/// passing it as one value is what stops the two halves from drifting apart
/// at a call site. This is the shape `&ExprArena` used to have by accident of
/// storing both; it is now the type.
#[derive(Clone, Copy)]
pub struct Term<'a> {
    root: Node<'a, ExprData>,
    env: &'a Environment,
}

impl<'a> Term<'a> {
    /// Pair a root with the environment its leaves index.
    #[must_use]
    pub fn new(root: Node<'a, ExprData>, env: &'a Environment) -> Self {
        Self { root, env }
    }

    /// The root node.
    #[must_use]
    pub fn root(self) -> Node<'a, ExprData> {
        self.root
    }

    /// The declaration tables.
    #[must_use]
    pub fn env(self) -> &'a Environment {
        self.env
    }

    /// The DAG the root belongs to.
    #[must_use]
    pub fn dag(self) -> &'a Dag<ExprData> {
        self.root.dag()
    }

    /// The same environment, re-rooted at `node`. `node` must come from this
    /// term's DAG.
    #[must_use]
    pub fn at(self, node: Node<'a, ExprData>) -> Self {
        Self {
            root: node,
            env: self.env,
        }
    }
}

// ────────────────────────────────────────── ExprBuilder ───────────────────────

/// A handle to a node under construction in an [`ExprBuilder`].
///
/// Opaque and builder-local: it can be spent on the builder that issued it and
/// nowhere else, and it does not survive [`ExprBuilder::finish`] — what crosses
/// that boundary is a [`Rooted<ExprData>`] whose nodes are named by [`Node`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ExprRef(u32);

/// Builds an expression graph and the environment its leaves index.
///
/// This is the public construction surface, and the only one: `dag::Builder`
/// is crate-private, because memory management is a `Dag`'s own business.
/// What is public here is the *expression* vocabulary — push a leaf, push an
/// op over children already pushed — plus declaration of the buffers and
/// uniforms those leaves name.
///
/// It also carries the [`Ir`](crate::term::Ir) implementation, which is why it
/// can read back what it has built ([`ExprBuilder::node`]): an optimizer that
/// destructures a node it just embedded needs that, and a write-only builder
/// could not offer it.
#[derive(Default)]
pub struct ExprBuilder {
    inner: Builder<ExprData>,
    /// `ExprRef` ordinal to the crate-private `Id` it stands for. The
    /// indirection is what keeps `dag::Id` sealed while still handing callers
    /// a `Copy + Ord` name for a node, which [`Ir::Ref`](crate::term::Ir::Ref)
    /// requires.
    ids: Vec<Id>,
    /// Children per node, in `ExprRef` terms, flat. The DAG names its own
    /// edges by `Id`, which cannot leave this crate, so
    /// [`Ir::project`](crate::term::Ir::project) — which must *lend* a child
    /// slice — has nowhere else to read them from.
    kid_slab: Vec<ExprRef>,
    kid_span: Vec<(u32, u32)>,
    env: Environment,
}

impl ExprBuilder {
    /// An empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn wrap(&mut self, id: Id, kids: &[ExprRef]) -> ExprRef {
        let r = ExprRef(self.ids.len() as u32);
        self.ids.push(id);
        self.kid_span
            .push((self.kid_slab.len() as u32, kids.len() as u32));
        self.kid_slab.extend_from_slice(kids);
        r
    }

    fn id(&self, r: ExprRef) -> Id {
        self.ids[r.0 as usize]
    }

    /// The node `r` names, for reading.
    #[must_use]
    pub fn node(&self, r: ExprRef) -> Node<'_, ExprData> {
        self.inner.get(self.id(r))
    }

    /// The children of the node `r` names, in operand order.
    #[must_use]
    pub fn child_refs(&self, r: ExprRef) -> &[ExprRef] {
        let (start, len) = self.kid_span[r.0 as usize];
        &self.kid_slab[start as usize..(start + len) as usize]
    }

    /// The declarations pushed so far.
    #[must_use]
    pub fn env(&self) -> &Environment {
        &self.env
    }

    /// How many nodes have been pushed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether nothing has been pushed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Freeze into a rooted DAG plus the environment its leaves index.
    #[must_use]
    pub fn finish(self, roots: &[ExprRef]) -> (Rooted<ExprData>, Environment) {
        let ids: Vec<Id> = roots.iter().map(|r| self.ids[r.0 as usize]).collect();
        (self.inner.finish(&ids), self.env)
    }

    // ───────────────────── leaves ──────────────────────

    /// Push a `Var(i)` node.
    ///
    /// Only `0..COORD_AXES` are lattice coordinates.
    /// [`RETIRED_COORD_AXES`] are the Z and W
    /// axes, which no longer exist: a graph that names one is refused where it
    /// would become code ([`Kernel::from_rooted`](crate::Kernel::from_rooted),
    /// and the JIT cache), rather than here, because the same node is also a
    /// reduction binder's index and a rewrite rule's pattern metavariable, and
    /// those namespaces are dense from zero.
    pub fn push_var(&mut self, i: u8) -> ExprRef {
        let id = self.inner.push_var(i);
        self.wrap(id, &[])
    }

    /// Push a `Const(v)` node.
    pub fn push_const(&mut self, v: f32) -> ExprRef {
        let id = self.inner.push_const(v);
        self.wrap(id, &[])
    }

    /// Push a `Param(i)` node.
    pub fn push_param(&mut self, i: u8) -> ExprRef {
        let id = self.inner.push_param(i);
        self.wrap(id, &[])
    }

    /// Declare a buffer slot, returning its [`BufferId`]. Declaring the same
    /// identity twice returns the slot it already has.
    pub fn declare_buffer(&mut self, decl: BufferDecl) -> BufferId {
        self.env.slot_for_buffer(decl)
    }

    /// Push a `Buffer(id)` leaf node.
    ///
    /// # Panics
    ///
    /// Panics if `id` has not been declared via [`ExprBuilder::declare_buffer`].
    pub fn push_buffer(&mut self, id: BufferId) -> ExprRef {
        assert!(
            (id.0 as usize) < self.env.buffers.len(),
            "push_buffer: BufferId({}) not declared (table has {} entries)",
            id.0,
            self.env.buffers.len()
        );
        let node = self.inner.push_buffer(id);
        self.wrap(node, &[])
    }

    /// Declare a uniform slot, returning its [`UniformId`]. Declaring the same
    /// identity twice returns the slot it already has.
    pub fn declare_uniform(&mut self, decl: UniformDecl) -> UniformId {
        self.env.slot_for_uniform(decl)
    }

    /// Push a `Uniform(id)` leaf node.
    ///
    /// # Panics
    ///
    /// Panics if `id` has not been declared via [`ExprBuilder::declare_uniform`].
    pub fn push_uniform(&mut self, id: UniformId) -> ExprRef {
        assert!(
            (id.0 as usize) < self.env.uniforms.len(),
            "push_uniform: UniformId({}) not declared (table has {} entries)",
            id.0,
            self.env.uniforms.len()
        );
        let node = self.inner.push_uniform(id);
        self.wrap(node, &[])
    }

    // ───────────────────── operators ───────────────────

    /// Push a unary operation node.
    pub fn push_unary(&mut self, op: OpKind, child: ExprRef) -> ExprRef {
        let a = self.id(child);
        let id = self.inner.push_unary(op, a);
        self.wrap(id, &[child])
    }

    /// Push a binary operation node.
    pub fn push_binary(&mut self, op: OpKind, a: ExprRef, b: ExprRef) -> ExprRef {
        let ids = (self.id(a), self.id(b));
        let id = self.inner.push_binary(op, ids.0, ids.1);
        self.wrap(id, &[a, b])
    }

    /// Push a ternary operation node.
    pub fn push_ternary(&mut self, op: OpKind, a: ExprRef, b: ExprRef, c: ExprRef) -> ExprRef {
        let ids = (self.id(a), self.id(b), self.id(c));
        let id = self.inner.push_ternary(op, ids.0, ids.1, ids.2);
        self.wrap(id, &[a, b, c])
    }

    /// Push an N-ary operation node.
    pub fn push_nary(&mut self, op: OpKind, children: &[ExprRef]) -> ExprRef {
        let kids: Vec<Id> = children.iter().map(|c| self.id(*c)).collect();
        let id = self.inner.push_nary(op, &kids);
        self.wrap(id, children)
    }

    /// Push a `Gather(buffer, x, y)` read of a declared buffer.
    ///
    /// Semantics: floor the indices, clamp to the declared extents, gather
    /// row-major. `DiscreteManifold::kernel` is exactly one of these.
    pub fn push_gather(&mut self, buffer: BufferId, x: ExprRef, y: ExprRef) -> ExprRef {
        let buf = self.push_buffer(buffer);
        self.push_ternary(OpKind::Gather, buf, x, y)
    }

    /// Push a reduction `Op(Reduce)` over `[Const(combiner), Const(binder),
    /// Const(extent), body]`.
    ///
    /// `combiner` is the monoid op folded with (`Add`/`Mul`/`Min`/`Max`/
    /// `BitAnd`/`BitOr`); `reduce_var` is the index the body folds over;
    /// `extent` is the trip count. Lowered to an unrolled accumulation by
    /// [`expand_reduce`](crate::passes::expand_reduce).
    ///
    /// # Panics
    ///
    /// Panics if `combiner` is not a monoid op or `reduce_var` is not a
    /// reduction binder index.
    pub fn push_reduce(
        &mut self,
        combiner: OpKind,
        reduce_var: u8,
        extent: u32,
        body: ExprRef,
    ) -> ExprRef {
        assert!(
            combiner.is_monoid(),
            "push_reduce: {combiner:?} is not a valid reduction combiner"
        );
        let binders = reduce_binders();
        assert!(
            binders.contains(&reduce_var),
            "push_reduce: reduce_var {reduce_var} out of range (must be {binders:?})",
        );
        let c = self.push_const(combiner.index() as f32);
        let v = self.push_const(f32::from(reduce_var));
        let n = self.push_const(extent as f32);
        self.push_nary(OpKind::Reduce, &[c, v, n, body])
    }

    // ───────────────────── copying in ──────────────────

    /// Copy `term`'s reachable subgraph in, merging its declarations into this
    /// builder's environment by identity — the composition primitive: reading
    /// the same memory (or the same uniform instance) from twenty places still
    /// binds one slot.
    pub fn splice(&mut self, term: Term<'_>) -> ExprRef {
        crate::term_dag::rebuild_into(term, self)
    }

    /// Copy `term`'s reachable subgraph in, replacing every `Param(i)` with
    /// what `params[i]` says it is: a `Const` folded into the fragment, or a
    /// `Uniform` slot declared for the handle's identity (one slot per
    /// identity, however many placeholders name it).
    ///
    /// # Panics
    ///
    /// Panics if any reachable `Param(i)` has `i >= params.len()`.
    pub fn substitute_params(&mut self, term: Term<'_>, params: &[Scalar]) -> ExprRef {
        let root = term.root();
        let mut table = root.dag().side_table(None);
        let mut stack = alloc::vec![(root, false)];
        while let Some((node, expanded)) = stack.pop() {
            if table[node].is_some() {
                continue;
            }
            if !expanded {
                stack.push((node, true));
                for child in node.children() {
                    if table[child].is_none() {
                        stack.push((child, false));
                    }
                }
                continue;
            }
            let r = match *node {
                ExprData::Param(i) => {
                    let scalar = params.get(i as usize).copied().unwrap_or_else(|| {
                        panic!(
                            "substitute_params: param index {i} out of range \
                             (have {} params)",
                            params.len()
                        )
                    });
                    match scalar {
                        Scalar::Const(v) => self.push_const(v),
                        Scalar::Uniform(u) => {
                            let slot = self.declare_uniform(u.decl());
                            self.push_uniform(slot)
                        }
                    }
                }
                ExprData::Var(i) => self.push_var(i),
                ExprData::Const(bits) => self.push_const(f32::from_bits(bits)),
                ExprData::Buffer(b) => {
                    let slot = self.declare_buffer(term.env().buffer(b));
                    self.push_buffer(slot)
                }
                ExprData::Uniform(u) => {
                    let slot = self.declare_uniform(term.env().uniform(u));
                    self.push_uniform(slot)
                }
                ExprData::Op(op) => {
                    let kids: Vec<ExprRef> = node
                        .children()
                        .map(|c| table[c].expect("child copied before parent"))
                        .collect();
                    self.push_nary(op, &kids)
                }
            };
            table[node] = Some(r);
        }
        table[root].expect("root must have been copied")
    }
}

// ────────────────────────────────────────── Splicing & Transforms ─────────────

/// Copy the reachable subgraph from `root` into `builder`, preserving DAG sharing.
pub(crate) fn copy_subgraph(builder: &mut Builder<ExprData>, root: Node<'_, ExprData>) -> Id {
    let mut table = root.dag().side_table(None);
    copy_subgraph_in(builder, root, &mut table)
}

/// Copy the reachable subgraph using an existing side table for memoization.
pub(crate) fn copy_subgraph_in(
    builder: &mut Builder<ExprData>,
    root: Node<'_, ExprData>,
    table: &mut SideTable<Option<Id>>,
) -> Id {
    let mut stack = alloc::vec![(root, false)];
    while let Some((node, expanded)) = stack.pop() {
        if table[node].is_some() {
            continue;
        }
        if expanded {
            let child_ids: Vec<Id> = node
                .children()
                .map(|c| table[c].expect("child must have been copied"))
                .collect();
            let id = builder.push_unique(*node, &child_ids);
            table[node] = Some(id);
        } else {
            stack.push((node, true));
            for child in node.children() {
                if table[child].is_none() {
                    stack.push((child, false));
                }
            }
        }
    }
    table[root].expect("root must have been copied")
}

/// Copy subgraph, replacing variables according to `subs`.
pub(crate) fn substitute_vars(
    builder: &mut Builder<ExprData>,
    root: Node<'_, ExprData>,
    subs: &[(u8, Id)],
) -> Id {
    let mut table = root.dag().side_table(None);
    let mut stack = alloc::vec![(root, false)];
    while let Some((node, expanded)) = stack.pop() {
        if table[node].is_some() {
            continue;
        }
        if expanded {
            let id = if let ExprData::Var(idx) = *node {
                if let Some(&(_, repl)) = subs.iter().find(|(v, _)| *v == idx) {
                    repl
                } else {
                    builder.push_unique(*node, &[])
                }
            } else {
                let child_ids: Vec<Id> = node
                    .children()
                    .map(|c| table[c].expect("child must have been copied"))
                    .collect();
                builder.push_unique(*node, &child_ids)
            };
            table[node] = Some(id);
        } else {
            stack.push((node, true));
            if !matches!(*node, ExprData::Var(_)) {
                for child in node.children() {
                    if table[child].is_none() {
                        stack.push((child, false));
                    }
                }
            }
        }
    }
    table[root].expect("root must have been copied")
}

/// Splicing: copy donor subgraph into `builder`, remapping buffer and uniform
/// slots into `env` by identity.
pub(crate) fn splice(
    builder: &mut Builder<ExprData>,
    env: &mut Environment,
    root: Node<'_, ExprData>,
    donor_env: &Environment,
) -> Id {
    let mut table = root.dag().side_table(None);
    let mut buf_map: Vec<Option<BufferId>> = alloc::vec![None; donor_env.buffers.len()];
    let mut uni_map: Vec<Option<UniformId>> = alloc::vec![None; donor_env.uniforms.len()];

    let mut stack = alloc::vec![(root, false)];
    while let Some((node, expanded)) = stack.pop() {
        if table[node].is_some() {
            continue;
        }
        if expanded {
            let data = match *node {
                ExprData::Buffer(b) => {
                    let slot = match buf_map[b.0 as usize] {
                        Some(s) => s,
                        None => {
                            let s = env.slot_for_buffer(donor_env.buffers[b.0 as usize]);
                            buf_map[b.0 as usize] = Some(s);
                            s
                        }
                    };
                    ExprData::Buffer(slot)
                }
                ExprData::Uniform(u) => {
                    let slot = match uni_map[u.0 as usize] {
                        Some(s) => s,
                        None => {
                            let s = env.slot_for_uniform(donor_env.uniforms[u.0 as usize]);
                            uni_map[u.0 as usize] = Some(s);
                            s
                        }
                    };
                    ExprData::Uniform(slot)
                }
                other => other,
            };
            let child_ids: Vec<Id> = node
                .children()
                .map(|c| table[c].expect("child must have been copied"))
                .collect();
            table[node] = Some(builder.push_unique(data, &child_ids));
        } else {
            stack.push((node, true));
            for child in node.children() {
                if table[child].is_none() {
                    stack.push((child, false));
                }
            }
        }
    }
    table[root].expect("root must have been copied")
}

// ────────────────────────────────────────── Linking ───────────────────────────

/// The subgraph reachable from `term`'s root, with its declaration tables
/// replaced by the given orders — the link step: slot `i` of the result names
/// `buffers[i]` / `uniforms[i]` and every reachable leaf is remapped to its
/// slot there.
///
/// Reachable nodes keep their relative order (ascending index, so still
/// topological), which is the order a schedule is built in; construction
/// garbage is dropped, which no schedule ever saw. So nothing downstream of
/// the tables — the schedule, the registers, the bytes — can move.
///
/// A declaration the term's environment holds but no reachable node reads is
/// left behind with the garbage: `Kernel::at` splices both coordinate
/// fragments whether or not the receiver reads that axis, so a table routinely
/// names an instance the graph does not, and the link — which is computed over
/// the reachable subgraph — rightly omits it. The orders may likewise declare
/// identities nothing here reads; those slots exist in the result unread.
///
/// # Panics
///
/// Panics if a *reachable* leaf's declaration has no entry in the given order,
/// or disagrees with it (extents, default).
#[must_use]
pub fn relink(
    term: Term<'_>,
    buffers: &[BufferDecl],
    uniforms: &[UniformDecl],
) -> (Rooted<ExprData>, Environment) {
    let dag = term.dag();
    let mut reachable = dag.side_table(false);
    for n in term.root().descendants() {
        reachable[n] = true;
    }

    let buffer_slot = |b: BufferId| -> BufferId {
        let decl = term.env().buffer(b);
        let i = buffers
            .iter()
            .position(|d| d.id == decl.id)
            .unwrap_or_else(|| panic!("relink: reachable {decl:?} is not in the link"));
        assert_eq!(buffers[i], decl, "relink: buffer declaration disagrees");
        BufferId(i as u16)
    };
    let uniform_slot = |u: UniformId| -> UniformId {
        let decl = term.env().uniform(u);
        let i = uniforms
            .iter()
            .position(|d| d.id == decl.id)
            .unwrap_or_else(|| panic!("relink: reachable {decl:?} is not in the link"));
        assert_eq!(uniforms[i], decl, "relink: uniform declaration disagrees");
        UniformId(i as u16)
    };

    let mut out: Builder<ExprData> = Builder::with_capacity(dag.len(), 0);
    let mut dense = dag.side_table(None);
    for node in dag.iter() {
        if !reachable[node] {
            continue;
        }
        let data = match *node {
            ExprData::Buffer(b) => ExprData::Buffer(buffer_slot(b)),
            ExprData::Uniform(u) => ExprData::Uniform(uniform_slot(u)),
            other => other,
        };
        let kids: Vec<Id> = node
            .children()
            .map(|c| dense[c].expect("relink: child densified before parent"))
            .collect();
        dense[node] = Some(out.push_unique(data, &kids));
    }

    let root = dense[term.root()].expect("relink: the root is reachable from itself");
    let env = Environment {
        buffers: buffers.to_vec(),
        uniforms: uniforms.to_vec(),
    };
    (out.finish(&[root]), env)
}

// ────────────────────────────────────────── Marshalling ───────────────────────

/// Format version written as the first byte of an [`encode`]d stream. Bumped
/// whenever the layout below changes, so a stale file fails loudly instead of
/// decoding into a different program.
const ENCODING_VERSION: u8 = 1;

const TAG_VAR: u8 = 0;
const TAG_CONST: u8 = 1;
const TAG_PARAM: u8 = 2;
const TAG_OP: u8 = 3;

/// Why a byte stream is not an expression graph.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecodeError {
    /// The stream ended in the middle of a record.
    Truncated,
    /// A version byte this build does not know how to read.
    Version(u8),
    /// A node tag naming no kind of node.
    Tag(u8),
    /// An op code naming no operation — a stale or corrupt stream.
    Op(u8),
    /// A child ordinal that is not a node already decoded. Children come
    /// strictly before parents, so a forward or out-of-range reference is not
    /// a DAG.
    Child(u32),
    /// A root ordinal past the end of the stream's nodes.
    Root(u32),
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => f.write_str("expression stream ended mid-record"),
            Self::Version(v) => write!(f, "unknown expression encoding version {v}"),
            Self::Tag(t) => write!(f, "unknown node tag {t}"),
            Self::Op(c) => write!(f, "op code {c} names no operation"),
            Self::Child(i) => write!(f, "child ordinal {i} is not an already-decoded node"),
            Self::Root(i) => write!(f, "root ordinal {i} is past the end"),
        }
    }
}

/// Marshal the subgraph reachable from `root` into a byte stream.
///
/// # Layout
///
/// Little-endian throughout. A `u8` format version (bumped whenever this layout
/// changes, so a stale file fails loudly instead of decoding into a different
/// program), a `u32` node
/// count, then that many node records, then a `u32` root ordinal. Records
/// appear in a topological order — **every child appears before its parent** —
/// and a child is named by its ordinal in that stream, so decoding is one
/// forward pass with no fixups. Node ordinals are dense over the reachable
/// subgraph: construction garbage does not travel.
///
/// A record is a tag byte and a payload:
///
/// | tag | node | payload |
/// |---|---|---|
/// | 0 | `Var(i)` | `u8` index |
/// | 1 | `Const(v)` | `u32` — the IEEE-754 **bit pattern**, so `-0.0` and every NaN survive |
/// | 2 | `Param(i)` | `u8` index |
/// | 3 | `Op(op)` | [`OpCode`] bytes, `u32` child count, then that many `u32` child ordinals |
///
/// # Panics
///
/// Panics on a reachable `Buffer` or `Uniform` leaf. Those name a
/// [`BufferDecl`]/[`UniformDecl`] in an [`Environment`], and a declaration is
/// a *binding*, not a value: its identity is minted per process, so writing
/// one down and reading it back in another process would hand two unrelated
/// blocks of memory the same name. Encode the expression, carry the
/// environment separately, or refuse — this refuses.
#[must_use]
pub fn encode(root: Node<'_, ExprData>) -> Vec<u8> {
    let mut out = Vec::new();
    encode_into(root, &mut out);
    out
}

/// [`encode`], appending to a buffer the caller owns.
pub fn encode_into(root: Node<'_, ExprData>, out: &mut Vec<u8>) {
    // Ordinals over the reachable subgraph, children before parents. The DAG's
    // own index order is already topological, so one ascending pass over the
    // reachable nodes assigns them.
    let dag = root.dag();
    let mut reachable = dag.side_table(false);
    for n in root.descendants() {
        reachable[n] = true;
    }
    let mut ordinal = dag.side_table(None);
    let mut records: Vec<Node<'_, ExprData>> = Vec::new();
    for node in dag.iter() {
        if !reachable[node] {
            continue;
        }
        ordinal[node] = Some(records.len() as u32);
        records.push(node);
    }

    out.push(ENCODING_VERSION);
    out.extend_from_slice(&(records.len() as u32).to_le_bytes());
    for node in &records {
        match **node {
            ExprData::Var(i) => {
                out.push(TAG_VAR);
                out.push(i);
            }
            ExprData::Const(bits) => {
                out.push(TAG_CONST);
                out.extend_from_slice(&bits.to_le_bytes());
            }
            ExprData::Param(i) => {
                out.push(TAG_PARAM);
                out.push(i);
            }
            ExprData::Buffer(b) => panic!(
                "encode: Buffer({}) is a binding, not a value — a BufferIdentity is \
                 minted per process and cannot be written down",
                b.0
            ),
            ExprData::Uniform(u) => panic!(
                "encode: Uniform({}) is a binding, not a value — a UniformIdentity is \
                 minted per process and cannot be written down",
                u.0
            ),
            ExprData::Op(op) => {
                out.push(TAG_OP);
                out.extend_from_slice(&op.marshal().to_bytes());
                out.extend_from_slice(&(node.child_count() as u32).to_le_bytes());
                for child in node.children() {
                    let ord = ordinal[child].expect("child ordinal assigned before parent");
                    out.extend_from_slice(&ord.to_le_bytes());
                }
            }
        }
    }
    let root_ord = ordinal[root].expect("the root is reachable from itself");
    out.extend_from_slice(&root_ord.to_le_bytes());
}

/// Rebuild an expression graph from bytes [`encode`] produced.
///
/// The result is single-entry: [`Rooted::entry`] is the node that was `root`.
/// It declares no buffers or uniforms, because the encoding cannot carry any
/// (see [`encode`]'s panic).
///
/// # Errors
///
/// Returns [`DecodeError`] for a truncated, stale or corrupt stream. Nothing
/// here trusts the bytes: an unknown op code, a forward child reference and a
/// root past the end are all refused rather than turned into a plausible
/// program.
pub fn decode(bytes: &[u8]) -> Result<Rooted<ExprData>, DecodeError> {
    let mut r = Reader { bytes, at: 0 };
    let version = r.u8()?;
    if version != ENCODING_VERSION {
        return Err(DecodeError::Version(version));
    }
    let count = r.u32()?;

    let mut b: Builder<ExprData> = Builder::with_capacity(count as usize, 0);
    let mut ids: Vec<Id> = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let id = match r.u8()? {
            TAG_VAR => b.push_var(r.u8()?),
            TAG_CONST => b.push_unique(ExprData::Const(r.u32()?), &[]),
            TAG_PARAM => b.push_param(r.u8()?),
            TAG_OP => {
                let code = r.u8()?;
                let op =
                    OpKind::unmarshal(OpCode::from_bytes([code])).ok_or(DecodeError::Op(code))?;
                let arity = r.u32()?;
                let mut kids: Vec<Id> = Vec::with_capacity(arity as usize);
                for _ in 0..arity {
                    let ord = r.u32()?;
                    kids.push(*ids.get(ord as usize).ok_or(DecodeError::Child(ord))?);
                }
                b.push_nary(op, &kids)
            }
            other => return Err(DecodeError::Tag(other)),
        };
        ids.push(id);
    }

    let root = r.u32()?;
    let root = *ids.get(root as usize).ok_or(DecodeError::Root(root))?;
    Ok(b.finish(&[root]))
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn u8(&mut self) -> Result<u8, DecodeError> {
        let b = *self.bytes.get(self.at).ok_or(DecodeError::Truncated)?;
        self.at += 1;
        Ok(b)
    }

    fn u32(&mut self) -> Result<u32, DecodeError> {
        let end = self.at + 4;
        let slice = self.bytes.get(self.at..end).ok_or(DecodeError::Truncated)?;
        self.at = end;
        Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
    }
}

// ────────────────────────────────────────── Inherent Node APIs ────────────────

impl<'a> Node<'a, ExprData> {
    /// Depth of the expression subtree rooted at `self`.
    #[must_use]
    pub fn depth(self) -> usize {
        // Bottom-up over the whole DAG rather than recursively per node: an
        // expression's depth is unbounded in principle (`Dwrt` chain-rule
        // expansion, deep composition), and a recursive walk of a 100k-deep
        // chain overflows the Rust stack.
        let mut table = self.dag().side_table(0usize);
        for n in self.dag().iter() {
            table[n] = 1 + n.children().map(|c| table[c]).max().unwrap_or(0);
        }
        table[self]
    }

    /// Whether any node in `self`'s reachable subgraph is a `Var`.
    #[must_use]
    pub fn has_var(self) -> bool {
        self.descendants().any(|x| matches!(*x, ExprData::Var(_)))
    }

    /// Whether `self`'s reachable subgraph contains a degenerate
    /// subexpression: a non-finite constant, `recip(0)`, or `div(_, 0)`.
    #[must_use]
    pub fn has_degenerate(self) -> bool {
        self.descendants().any(|x| match *x {
            ExprData::Const(b) => !f32::from_bits(b).is_finite(),
            ExprData::Op(OpKind::Recip) => x.children().any(is_const_zero),
            ExprData::Op(OpKind::Div) => x.children().nth(1).is_some_and(is_const_zero),
            _ => false,
        })
    }

    /// The retired coordinate axis reachable from `self`, if any — the guard
    /// [`Kernel::from_rooted`](crate::Kernel::from_rooted) and `emit::compile`
    /// apply before a graph can become a compiled kernel.
    ///
    /// A `Var(2)` reaching the emitter would read the third base coordinate,
    /// which a collapse passes as zero and no longer means anything: the
    /// pixels would be plausible and wrong. Refusing it is what makes "no
    /// emitted kernel reads the retired lanes" a fact rather than a habit.
    ///
    /// **Reachable from `self`, not every node in the DAG.** A transform
    /// leaves what it replaced behind, so a graph whose retired axes were
    /// correctly substituted still *holds* the original `Var(2)` nodes. They
    /// are not emitted, because nothing reaches them.
    #[must_use]
    pub fn retired_axis(self) -> Option<u8> {
        self.descendants().find_map(|x| match *x {
            ExprData::Var(i) if RETIRED_COORD_AXES.contains(&i) => Some(i),
            _ => None,
        })
    }

    /// Count unique nodes in `self`'s reachable subgraph.
    #[must_use]
    pub fn node_count(self) -> usize {
        self.descendants().count()
    }

    /// Structural equality between this subtree and `other`.
    ///
    /// Constants compare by bit pattern, so `-0.0` and `0.0` differ and a NaN
    /// equals itself. `Buffer`/`Uniform` leaves compare by **slot**: two
    /// graphs agreeing on slots but declaring different memory are equal here
    /// and distinguished by their environments, which is why
    /// [`Term::subtree_eq`] is the comparison to reach for when the tables
    /// might differ.
    #[must_use]
    pub fn subtree_eq(self, other: Node<'_, ExprData>) -> bool {
        // Iterative: expression depth is unbounded, so a recursive compare
        // would overflow on the deep chains `passes` builds.
        let mut stack = alloc::vec![(self, other)];
        while let Some((a, b)) = stack.pop() {
            if *a != *b || a.child_count() != b.child_count() {
                return false;
            }
            stack.extend(a.children().zip(b.children()));
        }
        true
    }
}

impl Term<'_> {
    /// Structural equality that also compares what the leaves *name*: two
    /// `Uniform(0)` leaves are equal only if slot 0 is the same instance in
    /// both environments.
    #[must_use]
    pub fn subtree_eq(self, other: Term<'_>) -> bool {
        let mut stack = alloc::vec![(self.root(), other.root())];
        while let Some((a, b)) = stack.pop() {
            if *a != *b || a.child_count() != b.child_count() {
                return false;
            }
            let declarations_agree = match (*a, *b) {
                (ExprData::Buffer(x), ExprData::Buffer(y)) => {
                    self.env().buffer(x) == other.env().buffer(y)
                }
                (ExprData::Uniform(x), ExprData::Uniform(y)) => {
                    self.env().uniform(x) == other.env().uniform(y)
                }
                _ => true,
            };
            if !declarations_agree {
                return false;
            }
            stack.extend(a.children().zip(b.children()));
        }
        true
    }
}

fn is_const_zero(n: Node<'_, ExprData>) -> bool {
    matches!(*n, ExprData::Const(b) if f32::from_bits(b) == 0.0)
}

// ────────────────────────────────────────── Traversal Functions ───────────────

/// Depth of the expression tree rooted at `n`.
#[must_use]
pub fn depth(n: Node<'_, ExprData>) -> usize {
    n.depth()
}

/// Bottom-up depth analysis for an entire DAG.
#[must_use]
pub fn compute_dag_depth(dag: &Dag<ExprData>) -> SideTable<usize> {
    let mut table = dag.side_table(0usize);
    for n in dag.iter() {
        table[n] = 1 + n.children().map(|c| table[c]).max().unwrap_or(0);
    }
    table
}

/// Whether any node in `n`'s reachable subgraph is a `Var`.
#[must_use]
pub fn has_var(n: Node<'_, ExprData>) -> bool {
    n.has_var()
}

/// Whether `n`'s reachable subgraph contains a degenerate subexpression.
#[must_use]
pub fn has_degenerate(n: Node<'_, ExprData>) -> bool {
    n.has_degenerate()
}

/// The retired coordinate axis (Z=2 or W=3) reachable from `n`, if any.
#[must_use]
pub fn retired_axis(n: Node<'_, ExprData>) -> Option<u8> {
    n.retired_axis()
}

/// Count unique nodes in `n`'s reachable subgraph.
#[must_use]
pub fn node_count_subtree(n: Node<'_, ExprData>) -> usize {
    n.node_count()
}

/// Structural equality between two expression subtrees.
#[must_use]
pub fn subtree_eq(a: Node<'_, ExprData>, b: Node<'_, ExprData>) -> bool {
    a.subtree_eq(b)
}

// ────────────────────────────────────────── Display ───────────────────────────

/// A [`Display`](core::fmt::Display)-able view of the subtree rooted at `n`,
/// as an S-expression: `mul_add(Var(0), Var(1), Const(2))`.
#[must_use]
pub fn display(n: Node<'_, ExprData>) -> DisplayExpr<'_> {
    DisplayExpr(n)
}

/// What [`display`] returns.
pub struct DisplayExpr<'a>(Node<'a, ExprData>);

impl core::fmt::Display for DisplayExpr<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Iterative: expression depth is unbounded, and a recursive formatter
        // would overflow on the deep chains `passes` builds.
        enum Task<'a> {
            Visit(Node<'a, ExprData>),
            Str(&'static str),
        }
        let mut stack = alloc::vec![Task::Visit(self.0)];
        while let Some(task) = stack.pop() {
            match task {
                Task::Str(s) => f.write_str(s)?,
                Task::Visit(n) => match *n {
                    ExprData::Var(i) => write!(f, "Var({i})")?,
                    ExprData::Const(b) => write!(f, "Const({})", f32::from_bits(b))?,
                    ExprData::Param(i) => write!(f, "Param({i})")?,
                    ExprData::Buffer(b) => write!(f, "Buffer({})", b.0)?,
                    ExprData::Uniform(u) => write!(f, "Uniform({})", u.0)?,
                    ExprData::Op(op) => {
                        stack.push(Task::Str(")"));
                        for (i, child) in n.children().enumerate().rev() {
                            stack.push(Task::Visit(child));
                            if i > 0 {
                                stack.push(Task::Str(", "));
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
}

// ────────────────────────────────────────── Tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Uniform;
    use crate::decl::{BufferDecl, BufferIdentity};
    use alloc::format;

    fn add_x_const(v: f32) -> (Rooted<ExprData>, Environment) {
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let c = b.push_const(v);
        let add = b.push_binary(OpKind::Add, x, c);
        b.finish(&[add])
    }

    #[test]
    fn builder_and_traversals() {
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let c = b.push_const(2.0);
        let mul = b.push_binary(OpKind::Mul, x, c);
        let add = b.push_binary(OpKind::Add, mul, x);
        let (rooted, env) = b.finish(&[add]);

        let root = rooted.entry();
        assert_eq!(depth(root), 3);
        assert!(has_var(root));
        assert!(!has_degenerate(root));
        assert_eq!(retired_axis(root), None);
        assert_eq!(node_count_subtree(root), 4); // add, mul, x, c (x shared)
        assert_eq!(root.op(), Some(OpKind::Add));
        assert!(env.buffers.is_empty());
    }

    #[test]
    fn degenerate_subexpressions_are_found() {
        for build in [
            |b: &mut ExprBuilder| b.push_const(f32::NAN),
            |b: &mut ExprBuilder| b.push_const(f32::INFINITY),
        ] {
            let mut b = ExprBuilder::new();
            let root = build(&mut b);
            let (r, _) = b.finish(&[root]);
            assert!(has_degenerate(r.entry()));
        }

        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let z = b.push_const(0.0);
        let div = b.push_binary(OpKind::Div, x, z);
        let (r, _) = b.finish(&[div]);
        assert!(has_degenerate(r.entry()));

        let mut b = ExprBuilder::new();
        let z = b.push_const(0.0);
        let recip = b.push_unary(OpKind::Recip, z);
        let (r, _) = b.finish(&[recip]);
        assert!(has_degenerate(r.entry()));

        // `div(0, x)` is fine: it is the DIVISOR that must be nonzero.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let z = b.push_const(0.0);
        let div = b.push_binary(OpKind::Div, z, x);
        let (r, _) = b.finish(&[div]);
        assert!(!has_degenerate(r.entry()));

        let (r, _) = add_x_const(1.0);
        assert!(!has_degenerate(r.entry()));
    }

    #[test]
    fn subtree_equality() {
        let (r1, _) = add_x_const(42.0);
        let (r2, _) = add_x_const(42.0);
        let (r3, _) = add_x_const(-42.0);
        assert!(subtree_eq(r1.entry(), r2.entry()));
        assert!(!subtree_eq(r1.entry(), r3.entry()));
    }

    #[test]
    fn a_term_comparison_distinguishes_uniform_instances() {
        let one = || {
            let mut b = ExprBuilder::new();
            let slot = b.declare_uniform(Uniform::new(1.0).decl());
            let u = b.push_uniform(slot);
            b.finish(&[u])
        };
        let (ra, ea) = one();
        let (rb, eb) = one();
        let a = Term::new(ra.entry(), &ea);
        let b = Term::new(rb.entry(), &eb);
        assert!(
            a.root().subtree_eq(b.root()),
            "same slot: structurally equal"
        );
        assert!(a.subtree_eq(a), "a term equals itself");
        assert!(!a.subtree_eq(b), "same slot, different instance");
    }

    #[test]
    fn copy_and_substitute() {
        let mut b1 = ExprBuilder::new();
        let x = b1.push_var(0);
        let c = b1.push_const(10.0);
        let root1 = b1.push_binary(OpKind::Mul, x, c);
        let (r1, e1) = b1.finish(&[root1]);

        let mut b2 = Builder::new();
        let y = b2.push_var(1);
        let subbed = substitute_vars(&mut b2, r1.entry(), &[(0, y)]);
        let r2 = b2.finish(&[subbed]);
        let _ = e1;

        let root2 = r2.entry();
        assert_eq!(root2.op(), Some(OpKind::Mul));
        let mut kids = root2.children();
        assert_eq!(*kids.next().unwrap(), ExprData::Var(1));
        assert_eq!(*kids.next().unwrap(), ExprData::constant(10.0));
    }

    #[test]
    fn splice_merges_environments_by_identity() {
        let u_decl = Uniform::new(3.5).decl();
        let donor = || {
            let mut b = ExprBuilder::new();
            let slot = b.declare_uniform(u_decl);
            let u = b.push_uniform(slot);
            let x = b.push_var(0);
            let root = b.push_binary(OpKind::Add, x, u);
            b.finish(&[root])
        };
        let (da, ea) = donor();
        let (db, eb) = donor();

        let mut target = ExprBuilder::new();
        let a = target.splice(Term::new(da.entry(), &ea));
        let b = target.splice(Term::new(db.entry(), &eb));
        let root = target.push_binary(OpKind::Mul, a, b);
        let (rooted, env) = target.finish(&[root]);

        assert_eq!(env.uniforms.len(), 1, "one identity, one slot");
        assert_eq!(env.uniforms[0], u_decl);
        assert_eq!(rooted.entry().op(), Some(OpKind::Mul));
    }

    #[test]
    fn substitute_params_folds_constants() {
        let mut b = ExprBuilder::new();
        let p0 = b.push_param(0);
        let p1 = b.push_param(1);
        let root = b.push_binary(OpKind::Add, p0, p1);
        let (r, e) = b.finish(&[root]);

        let mut out = ExprBuilder::new();
        let subbed = out.substitute_params(
            Term::new(r.entry(), &e),
            &[Scalar::Const(10.0), 20.0.into()],
        );
        let (rooted, _) = out.finish(&[subbed]);
        assert_eq!(
            format!("{}", display(rooted.entry())),
            "add(Const(10), Const(20))"
        );
    }

    #[test]
    fn display_is_an_s_expression() {
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let y = b.push_var(1);
        let c = b.push_const(2.0);
        let root = b.push_ternary(OpKind::MulAdd, x, y, c);
        let (r, _) = b.finish(&[root]);
        assert_eq!(
            format!("{}", display(r.entry())),
            "mul_add(Var(0), Var(1), Const(2))"
        );

        let mut b = ExprBuilder::new();
        let buf = b.declare_buffer(BufferDecl {
            id: BufferIdentity::mint(),
            width: 4,
            height: 1,
        });
        let leaf = b.push_buffer(buf);
        let (r, _) = b.finish(&[leaf]);
        assert_eq!(format!("{}", display(r.entry())), "Buffer(0)");
    }

    #[test]
    fn relink_densifies_and_renumbers() {
        let outer = Uniform::new(1.0).decl();
        let inner = Uniform::new(2.0).decl();
        let mut b = ExprBuilder::new();
        let s_outer = b.declare_uniform(outer);
        let s_inner = b.declare_uniform(inner);
        let _garbage = b.push_const(99.0);
        let u = b.push_uniform(s_inner);
        let x = b.push_var(0);
        let root = b.push_binary(OpKind::Add, x, u);
        let (r, env) = b.finish(&[root]);
        assert_eq!(s_outer, UniformId(0));

        // Link in the opposite order: the reachable leaf must be renumbered.
        let (linked, lenv) = relink(Term::new(r.entry(), &env), &[], &[inner, outer]);
        assert_eq!(lenv.uniforms, [inner, outer]);
        assert_eq!(
            linked.len(),
            3,
            "the unreachable Const does not survive the link"
        );
        let leaf = linked
            .iter()
            .find(|n| matches!(**n, ExprData::Uniform(_)))
            .expect("the uniform leaf survives");
        assert_eq!(*leaf, ExprData::Uniform(UniformId(0)));
    }

    #[test]
    #[should_panic(expected = "is not in the link")]
    fn relink_refuses_a_reachable_declaration_the_link_omits() {
        let mut b = ExprBuilder::new();
        let slot = b.declare_uniform(Uniform::new(1.0).decl());
        let root = b.push_uniform(slot);
        let (r, env) = b.finish(&[root]);
        let _refused = relink(Term::new(r.entry(), &env), &[], &[]);
    }

    // ───────────────────────── marshalling ─────────────────────────

    #[test]
    fn encoding_round_trips_structure_sharing_and_bit_patterns() {
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let s = b.push_binary(OpKind::Mul, x, x); // shared below
        let nan = b.push_const(f32::NAN);
        let neg_zero = b.push_const(-0.0);
        let p = b.push_param(7);
        let t = b.push_nary(OpKind::Tuple, &[s, s, nan, neg_zero, p]);
        let root = b.push_ternary(OpKind::MulAdd, s, x, t);
        let (r, _) = b.finish(&[root]);

        let bytes = encode(r.entry());
        let back = decode(&bytes).expect("round trip");

        assert_eq!(back.entry().node_count(), r.entry().node_count());
        assert!(back.entry().subtree_eq(r.entry()));
        // Sharing survives: `s` is one node, not two.
        assert_eq!(back.len(), r.entry().node_count());
        let consts: Vec<u32> = back
            .iter()
            .filter_map(|n| match *n {
                ExprData::Const(bits) => Some(bits),
                _ => None,
            })
            .collect();
        assert!(consts.contains(&f32::NAN.to_bits()), "NaN survives");
        assert!(consts.contains(&(-0.0f32).to_bits()), "-0.0 survives");
    }

    #[test]
    fn encoding_drops_unreachable_nodes() {
        let mut b = ExprBuilder::new();
        let _garbage = b.push_const(99.0);
        let x = b.push_var(0);
        let (r, _) = b.finish(&[x]);
        let back = decode(&encode(r.entry())).expect("round trip");
        assert_eq!(back.len(), 1);
        assert_eq!(*back.entry(), ExprData::Var(0));
    }

    #[test]
    #[should_panic(expected = "is a binding, not a value")]
    fn encoding_refuses_a_uniform() {
        let mut b = ExprBuilder::new();
        let slot = b.declare_uniform(Uniform::new(0.0).decl());
        let root = b.push_uniform(slot);
        let (r, _) = b.finish(&[root]);
        let _refused = encode(r.entry());
    }

    #[test]
    #[should_panic(expected = "is a binding, not a value")]
    fn encoding_refuses_a_buffer() {
        let mut b = ExprBuilder::new();
        let buf = b.declare_buffer(BufferDecl {
            id: BufferIdentity::mint(),
            width: 1,
            height: 1,
        });
        let root = b.push_buffer(buf);
        let (r, _) = b.finish(&[root]);
        let _refused = encode(r.entry());
    }

    #[test]
    fn decoding_refuses_corrupt_streams() {
        let (r, _) = add_x_const(1.0);
        let good = encode(r.entry());

        assert_eq!(decode(&[]).err(), Some(DecodeError::Truncated));
        for len in 1..good.len() {
            assert!(
                decode(&good[..len]).is_err(),
                "a stream cut at {len} must not decode"
            );
        }

        let mut stale = good.clone();
        stale[0] = ENCODING_VERSION.wrapping_add(1);
        assert_eq!(decode(&stale).err(), Some(DecodeError::Version(stale[0])));

        let mut bad_tag = good.clone();
        bad_tag[5] = 200;
        assert_eq!(decode(&bad_tag).err(), Some(DecodeError::Tag(200)));

        let mut bad_root = good.clone();
        let n = bad_root.len();
        bad_root[n - 4..].copy_from_slice(&9999u32.to_le_bytes());
        assert_eq!(decode(&bad_root).err(), Some(DecodeError::Root(9999)));
    }

    #[test]
    fn decoding_refuses_an_unknown_op_and_a_forward_child() {
        // Var, then an Op whose child ordinal points at itself.
        let mut bytes = alloc::vec![ENCODING_VERSION];
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&[TAG_VAR, 0]);
        bytes.push(TAG_OP);
        bytes.extend_from_slice(&OpKind::Neg.marshal().to_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes()); // itself — not yet decoded
        bytes.extend_from_slice(&1u32.to_le_bytes());
        assert_eq!(decode(&bytes).err(), Some(DecodeError::Child(1)));

        let mut bytes = alloc::vec![ENCODING_VERSION];
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.push(TAG_OP);
        bytes.push(u8::MAX);
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(decode(&bytes).err(), Some(DecodeError::Op(u8::MAX)));
    }
}
