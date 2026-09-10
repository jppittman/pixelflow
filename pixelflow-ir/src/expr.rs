//! Pure expression payload and DAG-centric expression APIs.
//!
//! Enforces the separation of concerns:
//! - **The DAG Structure** (`Dag<T>`): topology, acyclicity, ordering, and traversals.
//! - **The Node Payload** (`ExprData`): pure payload (`Var`, `Const`, `Param`, `Buffer`, `Uniform`, `Op`). Zero edges, zero child indices.
//! - **The Arena Shape / Environment**: pairs a `Rooted<ExprData>` with identity tables (`buffers`, `uniforms`).

use alloc::vec::Vec;

use crate::dag::{Builder, Dag, Id, Node, Rooted, SideTable};
use crate::declarations::{BufferDecl, BufferId, RETIRED_COORD_AXES, UniformDecl, UniformId};
use crate::fold::Fold;
use crate::kernel::Scalar;
use crate::key::KernelKey;
use crate::kind::OpKind;

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
    /// Bound variable: coordinate (0 for X, 1 for Y), reduction binder (4..8),
    /// or rewrite rule metavariable.
    Var(u8),
    /// 32-bit floating point literal stored as raw IEEE 754 bits.
    Const(u32),
    /// Macro parameter placeholder (0..). Valid only before kernel compilation.
    Param(u8),
    /// Memory buffer reference.
    Buffer(BufferId),
    /// Lattice-invariant scalar reference.
    Uniform(UniformId),
    /// Operator node. Arity (unary, binary, ternary, nary) is a property of the
    /// DAG edge count (`node.child_count()`).
    Op(OpKind),
    /// A bounded fold. Its one child is the body; everything else about it —
    /// monoid, binder, range — is [`Fold`], and lives here rather than in
    /// `Const` children.
    ///
    /// The alternative encoding is `Op(OpKind::Reduce)` over
    /// `[Const(combiner), Const(binder), Const(count), body]`, which is what
    /// this replaced. It costs a reader the question "is this float really a
    /// small integer, and is that integer really a binder slot?" — a question
    /// with no type-level answer, asked with `floorf` and a magic range, and
    /// re-asked by every pass. `Fold` answers it once, by construction, and
    /// `Fold::to_bits` gives the total order the cache keys need.
    ///
    /// One child, so "arity is a property of the edge count" still holds.
    Reduce(Fold),
    /// A kernel named rather than spliced — the content-addressed key a
    /// [`KernelStore`](crate::store) resolves.
    ///
    /// A leaf, like [`Buffer`](Self::Buffer) and [`Uniform`](Self::Uniform),
    /// and for the same reason: it names something this graph does not
    /// contain. What it names is a *kernel*, which is what makes it the one
    /// of the three that can be resolved back into graph.
    Ref(KernelKey),
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
    fn push_reduce(&mut self, fold: Fold, body: Id) -> Id;
    fn push_ref(&mut self, key: KernelKey) -> Id;
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

    #[inline]
    fn push_reduce(&mut self, fold: Fold, body: Id) -> Id {
        self.push_unique(ExprData::Reduce(fold), &[body])
    }

    #[inline]
    fn push_ref(&mut self, key: KernelKey) -> Id {
        self.push_unique(ExprData::Ref(key), &[])
    }
}

/// An opaque handle to a node under construction.
///
/// Handles cannot be inspected, forged, or mixed between builders. They are
/// consumed by [`ExprBuilder::finish`], where they become borrowed
/// [`Node`]s in the resulting rooted DAG.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ExprHandle(Id);

impl core::fmt::Debug for ExprHandle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("ExprHandle")
    }
}

/// The owned expression graph produced by [`ExprBuilder`].
///
/// The graph and its declaration environment travel together so a caller
/// never has to carry arena-local buffer or uniform slots beside expression
/// storage. Both are immutable after construction.
#[derive(Clone)]
pub struct ExprGraph {
    rooted: Rooted<ExprData>,
    env: Environment,
}

impl ExprGraph {
    /// Construct a graph from its owned representation pieces.
    #[must_use]
    pub fn new(rooted: Rooted<ExprData>, env: Environment) -> Self {
        Self { rooted, env }
    }

    /// The rooted expression DAG.
    #[must_use]
    pub fn rooted(&self) -> &Rooted<ExprData> {
        &self.rooted
    }

    /// The declaration environment for this graph.
    #[must_use]
    pub fn environment(&self) -> &Environment {
        &self.env
    }

    /// The graph's single expression root.
    #[must_use]
    pub fn root(&self) -> Node<'_, ExprData> {
        self.rooted.entry()
    }

    /// The graph's DAG structure.
    #[must_use]
    pub fn dag(&self) -> &Dag<ExprData> {
        &self.rooted
    }

    /// Split the graph into its immutable rooted DAG and environment.
    #[must_use]
    pub fn into_parts(self) -> (Rooted<ExprData>, Environment) {
        (self.rooted, self.env)
    }

    /// Replace every parameter leaf with a scalar argument.
    ///
    /// Constants become `Const` nodes; uniform arguments are declared in the
    /// returned environment and become `Uniform` leaves.  The source graph is
    /// unchanged, so one macro-produced template can safely build many
    /// kernels with different arguments.
    #[must_use]
    pub fn substitute_params(&self, params: &[Scalar]) -> Self {
        let mut builder = ExprBuilder {
            builder: Builder::new(),
            env: self.env.clone(),
        };
        let uniform_slots = params
            .iter()
            .map(|param| match param {
                Scalar::Const(_) => UniformId(0),
                Scalar::Uniform(uniform) => builder.env.slot_for_uniform(uniform.decl()),
            })
            .collect::<Vec<_>>();
        let root = substitute_params(&mut builder.builder, self.root(), params, &uniform_slots);
        ExprGraph::new(builder.builder.finish(&[root]), builder.env)
    }
}

/// Build-once expression construction API.
///
/// This is the only public construction surface for a new expression graph.
/// It keeps node handles opaque and records declarations by identity, leaving
/// the DAG's storage layout entirely private to `pixelflow-ir`.
pub struct ExprBuilder {
    builder: Builder<ExprData>,
    env: Environment,
}

impl Default for ExprBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprBuilder {
    /// Start an empty expression graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            builder: Builder::new(),
            env: Environment::new(),
        }
    }

    /// Start an expression graph with capacity for nodes and edges.
    #[must_use]
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            builder: Builder::with_capacity(nodes, edges),
            env: Environment::new(),
        }
    }

    #[inline]
    pub fn var(&mut self, var: u8) -> ExprHandle {
        ExprHandle(self.builder.intern(ExprData::Var(var), &[]))
    }

    #[inline]
    pub fn constant(&mut self, value: f32) -> ExprHandle {
        ExprHandle(self.builder.intern(ExprData::constant(value), &[]))
    }

    #[inline]
    pub fn param(&mut self, param: u8) -> ExprHandle {
        ExprHandle(self.builder.intern(ExprData::Param(param), &[]))
    }

    /// Add a buffer declaration and return the corresponding leaf.
    #[inline]
    pub fn buffer(&mut self, decl: BufferDecl) -> ExprHandle {
        let slot = self.env.slot_for_buffer(decl);
        ExprHandle(self.builder.intern(ExprData::Buffer(slot), &[]))
    }

    /// Add a uniform declaration and return the corresponding leaf.
    #[inline]
    pub fn uniform(&mut self, decl: UniformDecl) -> ExprHandle {
        let slot = self.env.slot_for_uniform(decl);
        ExprHandle(self.builder.intern(ExprData::Uniform(slot), &[]))
    }

    #[inline]
    pub fn reference(&mut self, key: KernelKey) -> ExprHandle {
        ExprHandle(self.builder.intern(ExprData::Ref(key), &[]))
    }

    #[inline]
    pub fn unary(&mut self, op: OpKind, child: ExprHandle) -> ExprHandle {
        ExprHandle(self.builder.intern(ExprData::Op(op), &[child.0]))
    }

    #[inline]
    pub fn binary(&mut self, op: OpKind, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        ExprHandle(self.builder.intern(ExprData::Op(op), &[a.0, b.0]))
    }

    #[inline]
    pub fn ternary(
        &mut self,
        op: OpKind,
        a: ExprHandle,
        b: ExprHandle,
        c: ExprHandle,
    ) -> ExprHandle {
        ExprHandle(self.builder.intern(ExprData::Op(op), &[a.0, b.0, c.0]))
    }

    #[inline]
    pub fn nary(&mut self, op: OpKind, children: &[ExprHandle]) -> ExprHandle {
        let ids: Vec<Id> = children.iter().map(|h| h.0).collect();
        ExprHandle(self.builder.intern(ExprData::Op(op), &ids))
    }

    #[inline]
    pub fn reduce(&mut self, fold: Fold, body: ExprHandle) -> ExprHandle {
        ExprHandle(self.builder.intern(ExprData::Reduce(fold), &[body.0]))
    }

    /// Freeze this builder into a graph with one expression root.
    #[must_use]
    pub fn finish_one(self, root: ExprHandle) -> ExprGraph {
        let ExprBuilder { builder, env } = self;
        ExprGraph::new(builder.finish(&[root.0]), env)
    }

    /// Freeze this builder into an immutable graph with the supplied roots.
    ///
    /// A builder may produce multiple roots; [`ExprGraph::root`] is available
    /// when exactly one root is supplied.
    #[must_use]
    pub fn finish(self, roots: &[ExprHandle]) -> ExprGraph {
        let ExprBuilder { builder, env } = self;
        let ids: Vec<Id> = roots.iter().map(|h| h.0).collect();
        ExprGraph::new(builder.finish(&ids), env)
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

/// Copy subgraph, replacing macro parameters with values.
///
/// Not yet called: `Kernel`'s parameter substitution still goes through the
/// legacy `ExprArena` path (`pixelflow-compiler/src/emit.rs`); this is the
/// `Dag`-native replacement staged for that, per
/// `docs/plans/2026-09-09-exprarena-on-dag.md`'s Stage C. `pub(crate)`
/// rather than `pub` made the gap visible (a `pub` fn is dead-code-exempt on
/// the assumption an external crate might call it, which none ever did) —
/// `#[allow(dead_code)]` because deleting or wiring this in isn't this
/// change's call to make.
#[allow(dead_code)]
pub(crate) fn substitute_params(
    builder: &mut Builder<ExprData>,
    root: Node<'_, ExprData>,
    params: &[Scalar],
    uniform_slots: &[UniformId],
) -> Id {
    let mut table = root.dag().side_table(None);
    let mut stack = alloc::vec![(root, false)];
    while let Some((node, expanded)) = stack.pop() {
        if table[node].is_some() {
            continue;
        }
        if expanded {
            let id = if let ExprData::Param(idx) = *node {
                match params.get(idx as usize) {
                    Some(Scalar::Const(v)) => builder.push_const(*v),
                    Some(Scalar::Uniform(_)) => {
                        let u_id = uniform_slots[idx as usize];
                        builder.push_uniform(u_id)
                    }
                    None => panic!("missing parameter substitution for slot {idx}"),
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
            if !matches!(*node, ExprData::Param(_)) {
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

// ─────────────────────────────────────── Arena Marshaling Boundary ────────────

// `ExprArena` is still the legacy wire representation while the rest of the
// workspace moves to `Dag`.  The conversion belongs to the representation that
// owns topology, however: callers ask a rooted DAG to unmarshal an old graph or
// ask a DAG node to marshal itself.  Keeping this here as free functions made
// the arena look like the primary representation and encouraged callers to
// traffic raw node ids alongside it.
impl Rooted<ExprData> {
    /// Unmarshal one or more legacy arena roots into an owned expression DAG.
    ///
    /// The returned environment owns the arena-local buffer and uniform slot
    /// declarations needed to interpret the expression payload.  The roots are
    /// kept in their supplied order.
    #[must_use]
    pub fn unmarshal(
        arena: &crate::arena::ExprArena,
        roots: &[crate::arena::ExprId],
    ) -> (Self, Environment) {
        use crate::arena::ExprNode;
        let mut b = Builder::new();
        let mut map: Vec<Option<Id>> = alloc::vec![None; arena.len()];

        enum Task {
            Descend(crate::arena::ExprId),
            Emit(crate::arena::ExprId),
        }
        let mut stack = Vec::new();
        for &r in roots.iter().rev() {
            stack.push(Task::Descend(r));
        }

        while let Some(task) = stack.pop() {
            match task {
                Task::Descend(id) => {
                    if map[id.0 as usize].is_some() {
                        continue;
                    }
                    stack.push(Task::Emit(id));
                    let children: Vec<crate::arena::ExprId> = arena.children(id).collect();
                    for c in children.into_iter().rev() {
                        stack.push(Task::Descend(c));
                    }
                }
                Task::Emit(id) => {
                    if map[id.0 as usize].is_some() {
                        continue;
                    }
                    let child_ids: Vec<Id> = arena
                        .children(id)
                        .map(|c| map[c.0 as usize].expect("child must be emitted before parent"))
                        .collect();
                    let new_id = match *arena.node(id) {
                        ExprNode::Var(i) => b.push_var(i),
                        ExprNode::Const(v) => b.push_const(v),
                        ExprNode::Param(i) => b.push_param(i),
                        ExprNode::Buffer(buf) => b.push_buffer(buf),
                        ExprNode::Uniform(uni) => b.push_uniform(uni),
                        ExprNode::Ref(key) => b.push_ref(key),
                        ExprNode::Reduce { fold, .. } => b.push_reduce(fold, child_ids[0]),
                        ExprNode::Unary(op, _)
                        | ExprNode::Binary(op, _, _)
                        | ExprNode::Ternary(op, _, _, _)
                        | ExprNode::Nary(op, _) => b.push_nary(op, &child_ids),
                    };
                    map[id.0 as usize] = Some(new_id);
                }
            }
        }

        let rooted_ids: Vec<Id> = roots
            .iter()
            .map(|&r| map[r.0 as usize].expect("root must be mapped"))
            .collect();
        let env = Environment {
            buffers: arena.buffers().to_vec(),
            uniforms: arena.uniforms().to_vec(),
        };
        (b.finish(&rooted_ids), env)
    }
}

impl<'a> Node<'a, ExprData> {
    /// Marshal this rooted expression into the legacy arena wire form.
    #[must_use]
    pub fn marshal(self, env: &Environment) -> (crate::arena::ExprArena, crate::arena::ExprId) {
        let (arena, roots) = self.dag().marshal(&[self], env);
        (arena, roots[0])
    }
}

impl Dag<ExprData> {
    /// Marshal expression roots into the legacy arena wire form.
    ///
    /// This is deliberately a DAG operation rather than an arena constructor:
    /// it preserves child-before-parent order and sharing while keeping the
    /// old storage layout confined to this compatibility boundary.
    #[must_use]
    pub fn marshal(
        &self,
        roots: &[Node<'_, ExprData>],
        env: &Environment,
    ) -> (crate::arena::ExprArena, Vec<crate::arena::ExprId>) {
        let mut arena = crate::arena::ExprArena::new();
        for b in &env.buffers {
            arena.declare_buffer(*b);
        }
        for u in &env.uniforms {
            arena.declare_uniform(*u);
        }

        let mut map = self.side_table(None);
        for node in self.iter() {
            let child_ids: Vec<crate::arena::ExprId> = node
                .children()
                .map(|c| map[c].expect("child already in arena"))
                .collect();
            let id = match *node {
                ExprData::Var(i) => arena.push_var(i),
                ExprData::Const(b) => arena.push_const(f32::from_bits(b)),
                ExprData::Param(i) => arena.push_param(i),
                ExprData::Buffer(b) => arena.push_buffer(b),
                ExprData::Uniform(u) => arena.push_uniform(u),
                ExprData::Ref(key) => arena.push_ref(key),
                ExprData::Reduce(fold) => arena.push_reduce(fold, child_ids[0]),
                ExprData::Op(op) => match child_ids.len() {
                    1 => arena.push_unary(op, child_ids[0]),
                    2 => arena.push_binary(op, child_ids[0], child_ids[1]),
                    3 => arena.push_ternary(op, child_ids[0], child_ids[1], child_ids[2]),
                    _ => arena.push_nary(op, &child_ids),
                },
            };
            map[node] = Some(id);
        }

        let root_ids = roots
            .iter()
            .map(|r| map[*r].expect("root must be mapped"))
            .collect();
        (arena, root_ids)
    }
}

// ────────────────────────────────────────── Environment & Splicing ────────────

/// Environment tables for buffers and uniforms.
#[derive(Clone, Default, Debug, PartialEq)]
pub struct Environment {
    pub buffers: Vec<BufferDecl>,
    pub uniforms: Vec<UniformDecl>,
}

impl Environment {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

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
                let id = BufferId(self.buffers.len() as u16);
                self.buffers.push(decl);
                id
            }
        }
    }

    pub fn slot_for_uniform(&mut self, decl: UniformDecl) -> UniformId {
        match self.uniforms.iter().position(|d| d.id == decl.id) {
            Some(i) => {
                assert_eq!(
                    self.uniforms[i], decl,
                    "two declarations share a UniformIdentity but disagree on default"
                );
                UniformId(i as u16)
            }
            None => {
                let id = UniformId(self.uniforms.len() as u16);
                self.uniforms.push(decl);
                id
            }
        }
    }
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
            let id = builder.push_unique(data, &child_ids);
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

// ────────────────────────────────────────── Inherent Node APIs ────────────────

impl<'a> Node<'a, ExprData> {
    /// Depth of the expression subtree rooted at `self`.
    #[must_use]
    pub fn depth(self) -> usize {
        1 + self.children().map(|c| c.depth()).max().unwrap_or(0)
    }

    /// Whether any node in `self`'s reachable subgraph is a `Var`.
    #[must_use]
    pub fn has_var(self) -> bool {
        self.descendants().any(|x| matches!(*x, ExprData::Var(_)))
    }

    /// Whether any constant in `self`'s reachable subgraph is NaN or infinity.
    #[must_use]
    pub fn has_degenerate(self) -> bool {
        self.descendants().any(|x| match *x {
            ExprData::Const(b) => {
                let v = f32::from_bits(b);
                v.is_nan() || v.is_infinite()
            }
            _ => false,
        })
    }

    /// Check if any retired coordinate axis (Z=2 or W=3) is referenced.
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
    #[must_use]
    pub fn subtree_eq(self, other: Node<'_, ExprData>) -> bool {
        if *self != *other || self.child_count() != other.child_count() {
            return false;
        }
        self.children()
            .zip(other.children())
            .all(|(ca, cb)| ca.subtree_eq(cb))
    }
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

/// Whether any constant in `n`'s reachable subgraph is NaN or infinity.
#[must_use]
pub fn has_degenerate(n: Node<'_, ExprData>) -> bool {
    n.has_degenerate()
}

/// Check if any retired coordinate axis (Z=2 or W=3) is referenced.
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

// ────────────────────────────────────────── Tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_ext_and_traversals() {
        let mut b = Builder::new();
        let x = b.push_var(0);
        let c = b.push_const(2.0);
        let mul = b.push_binary(OpKind::Mul, x, c);
        let add = b.push_binary(OpKind::Add, mul, x);
        let rooted = b.finish(&[add]);

        let root = rooted.entry();
        assert_eq!(depth(root), 3);
        assert!(has_var(root));
        assert!(!has_degenerate(root));
        assert_eq!(retired_axis(root), None);
        assert_eq!(node_count_subtree(root), 4); // add, mul, x, c (x shared)
    }

    #[test]
    fn subtree_equality() {
        let mut b1 = Builder::new();
        let x1 = b1.push_var(0);
        let c1 = b1.push_const(42.0);
        let add1 = b1.push_binary(OpKind::Add, x1, c1);
        let r1 = b1.finish(&[add1]);

        let mut b2 = Builder::new();
        let x2 = b2.push_var(0);
        let c2 = b2.push_const(42.0);
        let add2 = b2.push_binary(OpKind::Add, x2, c2);
        let r2 = b2.finish(&[add2]);

        assert!(subtree_eq(r1.entry(), r2.entry()));
    }

    #[test]
    fn copy_and_substitute() {
        let mut b1 = Builder::new();
        let x = b1.push_var(0);
        let c = b1.push_const(10.0);
        let root1 = b1.push_binary(OpKind::Mul, x, c);
        let r1 = b1.finish(&[root1]);

        let mut b2 = Builder::new();
        let y = b2.push_var(1);
        let subbed = substitute_vars(&mut b2, r1.entry(), &[(0, y)]);
        let r2 = b2.finish(&[subbed]);

        let root2 = r2.entry();
        assert_eq!(root2.op(), Some(OpKind::Mul));
        let mut kids = root2.children();
        let left = kids.next().unwrap();
        let right = kids.next().unwrap();
        assert_eq!(*left, ExprData::Var(1));
        assert_eq!(*right, ExprData::constant(10.0));
    }

    #[test]
    fn splice_merges_environment() {
        let mut donor_b = Builder::new();
        let mut donor_env = Environment::new();
        let u_decl = crate::Uniform::new(3.5).decl();
        let u_id = donor_env.slot_for_uniform(u_decl);
        let u_node = donor_b.push_uniform(u_id);
        let x = donor_b.push_var(0);
        let donor_root = donor_b.push_binary(OpKind::Add, x, u_node);
        let donor_rooted = donor_b.finish(&[donor_root]);

        let mut target_b = Builder::new();
        let mut target_env = Environment::new();
        let spliced_root = splice(
            &mut target_b,
            &mut target_env,
            donor_rooted.entry(),
            &donor_env,
        );
        let target_rooted = target_b.finish(&[spliced_root]);

        assert_eq!(target_env.uniforms.len(), 1);
        assert_eq!(target_env.uniforms[0], u_decl);
        let r = target_rooted.entry();
        assert_eq!(r.op(), Some(OpKind::Add));
    }

    #[test]
    fn marshal_unmarshal_roundtrip() {
        let mut arena = crate::arena::ExprArena::new();
        let x = arena.push_var(0);
        let c = arena.push_const(7.5);
        let add = arena.push_binary(OpKind::Add, x, c);

        let (rooted, env) = Rooted::unmarshal(&arena, &[add]);
        let root = rooted.entry();
        assert_eq!(root.op(), Some(OpKind::Add));
        assert_eq!(root.depth(), 2);

        let (arena2, root2) = root.marshal(&env);
        assert_eq!(arena2.depth(root2), 2);
    }

    #[test]
    fn public_builder_keeps_environment_with_graph() {
        let mut b = ExprBuilder::new();
        let x = b.var(0);
        let c = b.constant(2.0);
        let root = b.binary(OpKind::Mul, x, c);
        let graph = b.finish(&[root]);

        assert_eq!(graph.environment(), &Environment::new());
        assert_eq!(graph.root().op(), Some(OpKind::Mul));
        assert_eq!(graph.root().node_count(), 3);
    }

    #[test]
    #[should_panic(expected = "another DAG builder")]
    fn public_builder_rejects_handles_from_another_builder() {
        let mut first = ExprBuilder::new();
        let x = first.var(0);
        let mut second = ExprBuilder::new();
        let y = second.var(1);
        let _ = second.binary(OpKind::Add, x, y);
    }
}
