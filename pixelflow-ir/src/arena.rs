//! Arena-allocated expression storage.
//!
//! [`ExprArena`] is a [`Dag`](crate::dag::Dag) of expression data, indexed by
//! [`ExprId`] (a 4-byte Copy handle, translated to and from the DAG's own
//! opaque `Id` at this file's boundary — `ExprId`'s numeric value tracks the
//! DAG's own dense position, so the translation is free). This eliminates
//! per-node Arc overhead and gives O(1) `len()` for node counting.
//! docs/plans/2026-09-09-exprarena-on-dag.md is the staged migration that got
//! it here; [`crate::dag`]'s module doc has the design this file follows.
//!
//! Construction interns: `push_var`/`push_binary`/… structurally
//! hash-cons, so two pushes of the same value — same shape, same children —
//! return the same [`ExprId`]. Kernels are pure, so that is simply the right
//! answer, not a cache: nothing downstream of an arena can observe *how many*
//! times equal content was pushed, only what is reachable from a root. What a
//! caller must not assume any more is that a `push_*` call returns a *fresh*
//! id — see docs/plans/2026-09-09-exprarena-on-dag.md §5.2 for the callers
//! that assumption used to reach and how each was made safe under interning.
//!
//! The DAG's own storage is append-only and never deallocates before the
//! whole arena drops. [`ExprArena::clear`] truncates without deallocating,
//! ready for reuse.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::dag::{Builder, Id, Memo, Node};
use crate::fold::{Binder, Fold, Placeholder};
use crate::key::KernelKey;
use crate::kind::OpKind;

/// This arena's node payload: everything about a node except its edges,
/// which the underlying [`Dag`](crate::dag::Dag) owns. Distinct from
/// [`crate::expr::ExprData`], `Kernel`'s own Dag-native payload, which
/// collapses `Unary`/`Binary`/`Ternary`/`Nary` into one `Op(OpKind)` (arity
/// read off the edge count) because nothing there ever needs the arity
/// back: `Kernel` never constructs the same op two ways. `ExprArena` does —
/// `push_nary` is used for genuinely variable-arity content (`Tuple`) at
/// every arity including one, two and three — so collapsing here would
/// silently reclassify a small `Nary` as a `Unary`/`Binary`/`Ternary` the
/// moment it interned against (or merely came arity-first after) an
/// unrelated node of that shape, changing `kind`/`children`/`canonical`'s
/// tag byte for content nothing about the call site suggested would move.
/// One extra tag per arity is the price of keeping that impossible instead
/// of merely unlikely.
///
/// `Const` keys on the bit pattern, not the `f32`: `f32` is neither `Eq`
/// nor `Ord`, which the DAG's interning requires, and bits are what
/// [`ExprArena::subtree_eq`], the JIT cache key and the corpus format
/// already compare by — `-0.0` and `0.0` are different constants, and a NaN
/// payload is equal to itself.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
enum NodeData {
    Var(u8),
    Const(u32),
    Param(u8),
    Buffer(BufferId),
    Uniform(UniformId),
    Ref(KernelKey),
    Unary(OpKind),
    Binary(OpKind),
    Ternary(OpKind),
    Nary(OpKind),
    Reduce(Fold),
    /// `Write`'s three binders. The value is the node's one DAG edge, as
    /// for [`Reduce`](Self::Reduce) — see [`ExprNode::Write`]'s doc.
    Write(Binder, Binder, Binder),
}

/// [`ExprId`] ⇄ [`Id`]: `ExprId`'s numeric value *is* the DAG's own dense
/// position, so the translation costs nothing and needs no side table.
/// Free functions, not methods, because both directions are used before an
/// `ExprArena` is fully in scope (building the children slice to pass to
/// `Builder::intern`).
fn to_dag_id(id: ExprId) -> Id {
    Id::from_index(id.0)
}

fn from_dag_id(id: Id) -> ExprId {
    ExprId(id.index())
}

/// A node's DAG children, translated to [`ExprId`]s — the one place that
/// reads [`Node::children`] so [`ExprArena::node`]/[`ExprArena::children`]
/// do not each restate the translation.
fn dag_children(n: Node<'_, NodeData>) -> impl Iterator<Item = ExprId> + '_ {
    n.children()
        .map(|child| from_dag_id(Id::from_index(child.index())))
}

/// Coordinate axes a lattice has, and so the coordinate `Var` indices: `X = 0`,
/// `Y = 1`.
///
/// There were four. Z and W had extent 1 in every production call — an axis
/// that never varies is not an axis — so they left the language and the
/// scalars they carried became [`UniformDecl`]s
/// (docs/plans/2026-09-06-lattice-is-the-index.md).
pub const COORD_AXES: usize = 2;

/// One of the [`COORD_AXES`]: the `Var` a kernel reads a coordinate
/// through, and the axis a derivative is taken along.
///
/// The one numbering of the axes: `Kernel::x`/`y` and `kernel!`'s `X`/`Y`
/// read them by it, and a derivative names its axis by it
/// ([`library::derivative`](crate::library::derivative)).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Axis {
    /// `Var(0)`: the column.
    X = 0,
    /// `Var(1)`: the row.
    Y = 1,
}

impl Axis {
    /// Every axis, in `Var` order: axis `i` is `ALL[i]`.
    pub(crate) const ALL: [Self; COORD_AXES] = [Self::X, Self::Y];

    /// The `Var` index this axis is read through.
    #[must_use]
    pub const fn var(self) -> u8 {
        self as u8
    }
}

/// Why [`ExprArena::close_over`] built no fold: the body already binds every
/// [`Binder`], so the fold would be one deeper than the index space
/// ([`Binder::COUNT`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IndexSpaceFull;

/// A fold whose body is being built: what [`ExprArena::open_fold`] hands
/// out and [`ExprArena::close_fold`] takes back — the arena the fold is
/// built into, set aside while its body is built in a copy, and the
/// placeholder the body reads as its index.
///
/// Not `Clone`: one fold is closed once, and the arena it holds is put back
/// exactly then.
#[must_use = "an open fold is closed with `ExprArena::close_fold`, which restores the arena"]
pub struct OpenFold {
    enclosing: ExprArena,
    placeholder: Placeholder,
    index: ExprId,
}

impl OpenFold {
    /// The fold's index, as the body reads it: a placeholder `Var` until
    /// [`ExprArena::close_fold`] chooses the binder and renames it.
    #[must_use]
    pub fn index(&self) -> ExprId {
        self.index
    }
}

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
/// [`REDUCE_BINDER_BASE`] — the depth of nested folds a program may carry.
///
/// Every bit of a [`Variance`](crate::variance::Variance) past the
/// coordinates and the retired axes, and not a number chosen on its own:
/// the control plane is 64-bit, so the bitset is a `u64` and the binders
/// are what it has room for. It was four, sized to the deepest kernel then
/// written; the lattice's own three folds (docs/plans/2026-09-16-collapse-is-a-fold.md
/// §2.1) nest *outside* a kernel's, and would have left one for the kernel.
pub(crate) const REDUCE_BINDERS: u8 = u64::BITS as u8 - REDUCE_BINDER_BASE;

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
///
/// Still 32 bits, unlike [`UniformIdentity`]: buffers are leaving the
/// language (docs/plans/2026-09-25-the-language-is-kernel.md §1.6), so their
/// widths are left where they are rather than widened on the way out.
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
        static NEXT: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
        let id = u32::try_from(mint_identity(&NEXT, "BufferIdentity"))
            .unwrap_or_else(|_| panic!("BufferIdentity: counter exhausted"));
        Self(id)
    }
}

/// The one counter discipline behind every provenance identity, at the
/// control plane's width.
///
/// `fetch_add` + assert was wrong: the add WRAPS before the assert fires, so
/// if that panic is ever caught — or merely unwinds a non-fatal worker thread
/// — the counter has already returned to 0 and the next mint hands out an
/// identity that is still live. Two unrelated buffers (or uniforms) would
/// then compare identical and merge into one splice/JIT slot. `try_update`
/// declining to store leaves the counter permanently exhausted instead.
fn mint_identity(counter: &core::sync::atomic::AtomicU64, what: &str) -> u64 {
    counter
        .try_update(
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

/// Slot index into an [`ExprArena`]'s uniform table. Copy, 8 bytes. Not an
/// identity: two arenas each call their own first uniform slot 0.
///
/// 64 bits because nothing bounds how many arguments a program takes — a
/// glyph carries ten per piece and a font decides the piece count — and a
/// narrower slot was a limit on the language nobody chose (the `u16` it used
/// to be capped a kernel at 65,535 arguments through an assertion in
/// [`ExprArena::declare_uniform`]). The hardware's widths are applied where
/// the hardware sets them: the encoder that turns a slot into a
/// displacement.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct UniformId(pub u64);

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
pub struct UniformIdentity(u64);

impl UniformIdentity {
    /// Mint an identity distinct from every other in this process.
    ///
    /// # Panics
    ///
    /// Panics if the counter is exhausted, rather than wrapping onto a live
    /// identity and aliasing two unrelated uniforms.
    #[must_use]
    pub fn mint() -> Self {
        static NEXT: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
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

/// Where an [`ExprNode::Nary`] node's children live in
/// [`ExprArena`]'s n-ary slab.
///
/// Fields are private: this is storage, not expression semantics, and
/// docs/plans/2026-09-09-exprarena-on-dag.md's Stage B gate is exactly that
/// nothing outside this file can name an offset. A value can still be held
/// and passed around freely — it is `Copy` — but the only thing anything
/// outside `arena.rs` can do with one is match it as an opaque token; the
/// children it describes come back through [`ExprArena::children`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NaryChildren {
    start: u32,
    len: u16,
}

/// A single expression node stored in the arena.
///
/// Layout is kept tight: the static assertion below guarantees <= 16 bytes.
#[derive(Clone, Debug, PartialEq)]
pub enum ExprNode {
    /// A bound variable: a lattice coordinate ([`COORD_AXES`] of them, X and
    /// Y), a reduction binder's index (from [`REDUCE_BINDER_BASE`],
    /// [`REDUCE_BINDERS`] of them), or — in the macro front end
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
    /// [`link`](crate::passes::link) is what puts it back — its body as
    /// written ([`expand_refs`](crate::passes::expand_refs)), or as the runtime
    /// tier optimized it by itself (a *unit*).
    ///
    /// A leaf with an identity of its own, like [`ExprNode::Buffer`]: it has
    /// no children in *this* arena, and every pass that reads structure must
    /// expand it, refuse it, or hold it as an opaque leaf the way the runtime
    /// tier's e-graph holds a unit — never walk through it.
    Ref(KernelKey),
    Unary(OpKind, ExprId),
    Binary(OpKind, ExprId, ExprId),
    Ternary(OpKind, ExprId, ExprId, ExprId),
    /// N-ary node. Its children's location is private storage detail — see
    /// [`NaryChildren`] — and comes back through [`ExprArena::children`].
    Nary(OpKind, NaryChildren),
    /// A fold: `⊕_{k} body[fold.binder() := k]` over its range's visited
    /// indices (`lo`, `lo+stride`, …, [`Fold::len`] of them) — see [`Fold`].
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
    /// A store: the one effect in the language, and the body of the folds a
    /// lattice is (docs/plans/2026-09-16-collapse-is-a-fold.md §2.4).
    ///
    /// `⟦Write { row, col, lane, value }⟧` stores `value`'s first `len(lane)`
    /// lanes at `out + 4·(row·pitch + col + lane)`, where `out` and `pitch`
    /// are the call's arguments — the collapse ABI is `fn(ctx, out, pitch)`,
    /// with one output plane, so the node names nothing about *where* the
    /// plane is. It names **binders**, not an address expression: `row`,
    /// `col` and `lane` are the three lattice folds' indices, so contiguity
    /// along `lane` holds by construction and needs no analysis, and the
    /// store's width is the lane fold's trip count rather than a field here.
    ///
    /// Unit-typed: its value is nothing, which is why the folds it sits in
    /// are over [`Monoid::SEQ`](crate::fold::Monoid::SEQ). Constructible only
    /// by the legalize passes that wrap a kernel in the lattice
    /// ([`ExprArena::push_write`] is `pub(crate)`); the e-graph declines one
    /// and `kernel!` cannot name one.
    /// One child, the value, as `Reduce` has its body.
    Write {
        row: Binder,
        col: Binder,
        lane: Binder,
        value: ExprId,
    },
}

// A tripwire against an accident, **not** a design constraint, and the
// distinction is the point: nothing in this workspace depends on a node's
// width — no serialization format, no fixed-width record, no mapped file, no
// alignment requirement beyond what `KernelKey` already forces. Assert a
// bound so that boxing something large, or storing a `String`, fails a build
// instead of quietly costing every node in every kernel. Do not read it as a
// budget to design against.
//
// It used to be 16, which was not chosen either — it was whatever `Ref`'s one
// `KernelKey` happened to need, recorded as though it were a requirement.
// That is how `Fold` came to carry `u16` ends "so the two fit the node in the
// 16 bytes `ExprNode` is capped at", making a self-imposed width into a cap
// on how many terms a reduction may have. A number nothing depends on should
// never propagate into the language's semantics, so this one is now loose
// enough that adding a node is not a conversation about bytes.
const _: () = assert!(
    core::mem::size_of::<ExprNode>() <= 32,
    "ExprNode must fit in 32 bytes"
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

// From the back as well, for a walk that pushes children onto a stack and
// wants to pop the first one first — `key::canonical`'s post-order — without
// collecting them into a `Vec` at every node on the way.
impl DoubleEndedIterator for ExprChildren<'_> {
    fn next_back(&mut self) -> Option<ExprId> {
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
                *self = Self::One(a);
                Some(b)
            }
            Self::Three(a, b, c) => {
                let a = *a;
                let b = *b;
                let c = *c;
                *self = Self::Two(a, b);
                Some(c)
            }
            Self::Nary(slice) => {
                let (last, rest) = slice.split_last()?;
                *self = Self::Nary(rest);
                Some(*last)
            }
        }
    }
}

// ───────────────────────────────────── ExprArena ─────────────────────────────

/// Arena-allocated expression storage. Append-only, O(1) drop.
///
/// A [`Builder<NodeData>`](crate::dag::Builder) rather than a
/// [`Dag`](crate::dag::Dag): unlike `Kernel`, which builds once per
/// combinator call and freezes into a [`Rooted`](crate::dag::Rooted), an
/// `ExprArena` is grown by an unbounded number of `push_*` calls over its
/// whole lifetime — every combinator, every rewrite, every corpus read —
/// so it needs the always-growable builder, never the frozen shape.
#[derive(Clone)]
pub struct ExprArena {
    builder: Builder<NodeData>,
    /// The append-only cache backing every interned [`ExprNode::Nary`]
    /// node's children: [`NaryChildren`] is only ever a range into this.
    /// Not derived from `builder`'s own edges on demand, because
    /// `ExprChildren::Nary` hands out a borrowed `&[ExprId]` — the same
    /// contract it had before this file's storage changed — and there is
    /// nowhere to borrow one from a freshly-computed set of edges. Grown
    /// exactly once per *distinct* `Nary` node ([`ExprArena::push_nary`]):
    /// interning a repeat is a no-op here too, via [`Self::nary_ranges`].
    nary_children: Vec<ExprId>,
    /// Which slice of [`Self::nary_children`] each `Nary` node at this
    /// dense position owns, filled in the first time that node is
    /// interned. `None` for every other position — most of them, since
    /// only `Nary` nodes have an entry at all.
    nary_ranges: Vec<Option<NaryChildren>>,
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
            builder: Builder::new(),
            nary_children: Vec::new(),
            nary_ranges: Vec::new(),
            buffers: Vec::new(),
            uniforms: Vec::new(),
        }
    }

    /// Create an arena pre-allocated for `n` nodes.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            builder: Builder::with_capacity(n, n),
            nary_children: Vec::new(),
            nary_ranges: Vec::new(),
            buffers: Vec::new(),
            uniforms: Vec::new(),
        }
    }

    /// Truncate to zero nodes without deallocating backing storage.
    pub fn clear(&mut self) {
        self.builder.clear();
        self.nary_children.clear();
        self.nary_ranges.clear();
        self.buffers.clear();
        self.uniforms.clear();
    }

    /// Number of nodes in the arena. This is the O(1) node count.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.builder.dag().len()
    }

    /// Returns `true` if the arena contains no nodes.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.builder.dag().is_empty()
    }

    // ───────────────────── push helpers ──────────────────────

    /// Intern `data` with `children`, translating to and from this file's
    /// [`ExprId`] boundary — the one choke point every `push_*` method
    /// spends its arguments through. Structural: two pushes of equal `data`
    /// over equal `children` return the same id (see the module doc).
    fn intern(&mut self, data: NodeData, children: &[ExprId]) -> ExprId {
        let dag_children: Vec<Id> = children.iter().copied().map(to_dag_id).collect();
        from_dag_id(self.builder.intern(data, &dag_children))
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
        self.intern(NodeData::Var(i), &[])
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
        let mut seen = alloc::vec![false; self.len()];
        let mut stack = alloc::vec![root];
        while let Some(id) = stack.pop() {
            let idx = id.0 as usize;
            if core::mem::replace(&mut seen[idx], true) {
                continue;
            }
            if let ExprNode::Var(i) = &self.node(id)
                && RETIRED_COORD_AXES.contains(i)
            {
                return Some(*i);
            }
            stack.extend(self.children(id));
        }
        None
    }

    /// The lowest index `root` reads that no fold within it binds — a
    /// reduction binder's `Var`, or a placeholder's — or `None` when `root`
    /// is **closed**: every index it reads is bound by a `Reduce` around the
    /// read, inside `root`.
    ///
    /// `Var`'s index space is three namespaces stacked in one integer —
    /// coordinates, then the reserved retired axes, then reduction binders,
    /// then a binder's under-construction placeholder. A coordinate is read,
    /// never bound, and is not an index; everything from the first binder up
    /// is one, and a term that reads one it does not bind means nothing on
    /// its own: its value depends on a fold outside it. That is the question
    /// a name ([`Kernel::by_ref`](crate::Kernel::by_ref)) and a kernel-typed
    /// argument ([`ExprArena::admit`]) both ask, so it is asked here once.
    ///
    /// Scoped, not a scan: a binder read under the `Reduce` that binds it is
    /// bound, and the same `Var` read outside it is free. Asking only whether
    /// some `Var` above a floor is reachable — what this replaced — saw a
    /// placeholder and missed a free binder, which the next fold built around
    /// the term could then choose, and capture.
    ///
    /// Reachable from `root`, not every node, for
    /// [`retired_axis`](ExprArena::retired_axis)'s reason: an arena keeps the
    /// nodes a rebuild replaced, and nothing evaluates those.
    #[must_use]
    pub fn free_index(&self, root: ExprId) -> Option<u8> {
        /// One bit per `Var` index a `u8` can name.
        type Indices = [u128; 2];
        fn bit(index: u8) -> Indices {
            let mut indices = [0; 2];
            indices[usize::from(index / 128)] = 1 << (index % 128);
            indices
        }
        let mut reached = alloc::vec![false; root.0 as usize + 1];
        let mut stack = alloc::vec![root];
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut reached[id.0 as usize], true) {
                continue;
            }
            stack.extend(self.children(id));
        }
        // Children precede their parents in an arena, so one ascending pass
        // sees every child's free indices before its parent asks for them.
        let mut free: Vec<Indices> = alloc::vec![[0; 2]; root.0 as usize + 1];
        for index in (0..=root.0 as usize).filter(|&index| reached[index]) {
            let id = ExprId(index as u32);
            free[index] = match self.node(id) {
                ExprNode::Var(i) if i >= REDUCE_BINDER_BASE => bit(i),
                ExprNode::Reduce { fold, body } => {
                    let bound = bit(fold.binder().var());
                    let [low, high] = free[body.0 as usize];
                    [low & !bound[0], high & !bound[1]]
                }
                _ => self.children(id).fold([0; 2], |[low, high], child| {
                    let [child_low, child_high] = free[child.0 as usize];
                    [low | child_low, high | child_high]
                }),
            };
        }
        match free[root.0 as usize] {
            [0, 0] => None,
            [0, high] => Some(128 + high.trailing_zeros() as u8),
            [low, _] => Some(low.trailing_zeros() as u8),
        }
    }

    /// Push a `Const(v)` node.
    pub fn push_const(&mut self, v: f32) -> ExprId {
        self.intern(NodeData::Const(v.to_bits()), &[])
    }

    /// Push a `Param(i)` node.
    pub fn push_param(&mut self, i: u8) -> ExprId {
        self.intern(NodeData::Param(i), &[])
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
        self.intern(NodeData::Buffer(id), &[])
    }

    /// Declare a uniform slot, returning its [`UniformId`]. The table has no
    /// cap of its own: a slot is 64 bits wide, and how many arguments a
    /// program takes is the program's business.
    pub fn declare_uniform(&mut self, decl: UniformDecl) -> UniformId {
        let id = UniformId(self.uniforms.len() as u64);
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
                UniformId(i as u64)
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
        self.intern(NodeData::Uniform(id), &[])
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
        self.intern(NodeData::Ref(key), &[])
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

    /// Push the fold `fold` performs over `body` — a `⊕` over a range.
    ///
    /// Two arguments, because [`Fold`] is the metadata: which algebra, which
    /// index, which range. Every one of those was an assertion here — a
    /// combiner that is a monoid, a var index inside the binder space, a trip
    /// count that fits — and each is now a thing the type will not build.
    /// A survivor reaches codegen as a loop.
    pub fn push_reduce(&mut self, fold: Fold, body: ExprId) -> ExprId {
        self.intern(NodeData::Reduce(fold), &[body])
    }

    /// A store of `value` at the lattice position the three binders name —
    /// see [`ExprNode::Write`].
    ///
    /// Crate-private: only the legalize passes that wrap a kernel in the
    /// lattice's folds may build one. Nothing a `Kernel` can say, and
    /// nothing the e-graph will hold.
    pub(crate) fn push_write(
        &mut self,
        row: Binder,
        col: Binder,
        lane: Binder,
        value: ExprId,
    ) -> ExprId {
        self.intern(NodeData::Write(row, col, lane), &[value])
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
        self.intern(NodeData::Unary(op), &[child])
    }

    /// Push a binary operation node.
    pub fn push_binary(&mut self, op: OpKind, a: ExprId, b: ExprId) -> ExprId {
        self.intern(NodeData::Binary(op), &[a, b])
    }

    /// Push a ternary operation node.
    pub fn push_ternary(&mut self, op: OpKind, a: ExprId, b: ExprId, c: ExprId) -> ExprId {
        self.intern(NodeData::Ternary(op), &[a, b, c])
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
        let id = self.intern(NodeData::Nary(op), children);
        // Interning a repeat returns an id this table already has an entry
        // for — the range recorded the first time this exact (op, children)
        // was pushed. Only a genuinely new node needs one grown.
        let idx = id.0 as usize;
        if self.nary_ranges.len() <= idx {
            self.nary_ranges.resize(idx + 1, None);
        }
        if self.nary_ranges[idx].is_none() {
            let start = self.nary_children.len() as u32;
            let len = children.len() as u16;
            self.nary_children.extend_from_slice(children);
            self.nary_ranges[idx] = Some(NaryChildren { start, len });
        }
        id
    }

    // ───────────────────── node observation ──────────────────

    /// Every node with its id, in construction order — children strictly
    /// before parents, since the arena is append-only and a node may only
    /// reference an id less than its own.
    ///
    /// This is the topological order every "scan every node" pass already
    /// relies on. A caller that wants a node's edges goes through
    /// [`ExprArena::children`] instead, never through the n-ary slab
    /// directly — that slab, and its offsets, are `arena.rs`'s own business
    /// (docs/plans/2026-09-09-exprarena-on-dag.md, Stage B: nothing outside
    /// this file names an offset, which is why there is no `nodes_raw`/
    /// `nary_children_raw` pair here any more).
    #[inline]
    pub fn nodes(&self) -> impl DoubleEndedIterator<Item = (ExprId, ExprNode)> + '_ {
        (0..self.builder.dag().len() as u32).map(move |i| {
            let id = ExprId(i);
            (id, self.node(id))
        })
    }

    // ───────────────────── access ────────────────────────────

    /// Reconstruct the node at `id`.
    ///
    /// Owned, not borrowed: there is no `Vec<ExprNode>` behind this arena
    /// any more to hold a reference into (docs/plans/2026-09-09-exprarena-
    /// on-dag.md, Stage C) — every call rebuilds an [`ExprNode`] from this
    /// node's [`NodeData`] and its DAG edges. Every field `ExprNode` can
    /// hold is `Copy`, so a caller that used to match `arena.node(id)` by
    /// reference sees the same bindings matching the owned value directly.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of bounds.
    #[inline]
    #[must_use]
    pub fn node(&self, id: ExprId) -> ExprNode {
        let n = self.builder.dag().get(id.0);
        match *n {
            NodeData::Var(i) => ExprNode::Var(i),
            NodeData::Const(bits) => ExprNode::Const(f32::from_bits(bits)),
            NodeData::Param(i) => ExprNode::Param(i),
            NodeData::Buffer(b) => ExprNode::Buffer(b),
            NodeData::Uniform(u) => ExprNode::Uniform(u),
            NodeData::Ref(k) => ExprNode::Ref(k),
            NodeData::Unary(op) => {
                let mut kids = dag_children(n);
                let a = kids.next().expect("a Unary node has one DAG child");
                ExprNode::Unary(op, a)
            }
            NodeData::Binary(op) => {
                let mut kids = dag_children(n);
                let a = kids.next().expect("a Binary node has two DAG children");
                let b = kids.next().expect("a Binary node has two DAG children");
                ExprNode::Binary(op, a, b)
            }
            NodeData::Ternary(op) => {
                let mut kids = dag_children(n);
                let a = kids.next().expect("a Ternary node has three DAG children");
                let b = kids.next().expect("a Ternary node has three DAG children");
                let c = kids.next().expect("a Ternary node has three DAG children");
                ExprNode::Ternary(op, a, b, c)
            }
            NodeData::Nary(op) => {
                let range = self.nary_ranges[id.0 as usize]
                    .expect("a Nary node's range is recorded when it is first interned");
                ExprNode::Nary(op, range)
            }
            NodeData::Reduce(fold) => {
                let mut kids = dag_children(n);
                let body = kids.next().expect("a Reduce node has one DAG child");
                ExprNode::Reduce { fold, body }
            }
            NodeData::Write(row, col, lane) => {
                let mut kids = dag_children(n);
                let value = kids.next().expect("a Write node has one DAG child");
                ExprNode::Write {
                    row,
                    col,
                    lane,
                    value,
                }
            }
        }
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
        match *self.builder.dag().get(id.0) {
            NodeData::Var(_) => OpKind::Var,
            NodeData::Const(_) | NodeData::Param(_) => OpKind::Const,
            NodeData::Buffer(_) => OpKind::Buffer,
            NodeData::Uniform(_) => OpKind::Uniform,
            NodeData::Ref(key) => panic!(
                "ExprArena::kind: {key:?} is a reference to a kernel, not an \
                 operation; run passes::expand_refs before asking for a kind"
            ),
            NodeData::Unary(op)
            | NodeData::Binary(op)
            | NodeData::Ternary(op)
            | NodeData::Nary(op) => op,
            NodeData::Reduce(_) => OpKind::Reduce,
            // A store is an effect, not an operation: it computes nothing a
            // cost table could price or an arithmetic rule could rewrite,
            // and the one consumer that executes it (the emitter, on the
            // folds a lattice is) reads the node, not a kind.
            NodeData::Write(..) => panic!(
                "ExprArena::kind: a Write is a store, not an operation — no \
                 OpKind describes it; read the node's value and binders directly"
            ),
        }
    }

    /// Iterate over the child [`ExprId`]s of the node at `id`.
    #[inline]
    #[must_use]
    pub fn children(&self, id: ExprId) -> ExprChildren<'_> {
        let n = self.builder.dag().get(id.0);
        match *n {
            NodeData::Var(_)
            | NodeData::Const(_)
            | NodeData::Param(_)
            | NodeData::Buffer(_)
            | NodeData::Uniform(_)
            | NodeData::Ref(_) => ExprChildren::Zero,
            NodeData::Unary(_) => {
                let mut kids = dag_children(n);
                ExprChildren::One(kids.next().expect("a Unary node has one DAG child"))
            }
            NodeData::Binary(_) => {
                let mut kids = dag_children(n);
                let a = kids.next().expect("a Binary node has two DAG children");
                let b = kids.next().expect("a Binary node has two DAG children");
                ExprChildren::Two(a, b)
            }
            NodeData::Ternary(_) => {
                let mut kids = dag_children(n);
                let a = kids.next().expect("a Ternary node has three DAG children");
                let b = kids.next().expect("a Ternary node has three DAG children");
                let c = kids.next().expect("a Ternary node has three DAG children");
                ExprChildren::Three(a, b, c)
            }
            NodeData::Nary(_) => {
                let range = self.nary_ranges[id.0 as usize]
                    .expect("a Nary node's range is recorded when it is first interned");
                let s = range.start as usize;
                let l = range.len as usize;
                ExprChildren::Nary(&self.nary_children[s..s + l])
            }
            // One child, not four: the combiner, the binder and the extent
            // are no longer expressions, so nothing that walks children can
            // reach them, fold them, or cost them.
            NodeData::Reduce(_) => {
                let mut kids = dag_children(n);
                ExprChildren::One(kids.next().expect("a Reduce node has one DAG child"))
            }
            // One child, the value stored. The binders are the node's own
            // metadata, as a `Reduce`'s fold is: an index, not an operand.
            NodeData::Write(..) => {
                let mut kids = dag_children(n);
                ExprChildren::One(kids.next().expect("a Write node has one DAG child"))
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
            match &self.node(id) {
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
                ExprNode::Nary(_, range) => {
                    let s = range.start as usize;
                    let l = range.len as usize;
                    if l == 0 {
                        max_depth = max_depth.max(d);
                    } else {
                        for child in &self.nary_children[s..s + l] {
                            stack.push((*child, d + 1));
                        }
                    }
                }
                ExprNode::Reduce { body, .. } => stack.push((*body, d + 1)),
                ExprNode::Write { value, .. } => stack.push((*value, d + 1)),
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
            match &self.node(id) {
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
                ExprNode::Nary(_, range) => {
                    let s = range.start as usize;
                    let l = range.len as usize;
                    for child in &self.nary_children[s..s + l] {
                        stack.push(*child);
                    }
                }
                ExprNode::Reduce { body, .. } => stack.push(*body),
                ExprNode::Write { value, .. } => stack.push(*value),
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
            match &self.node(id) {
                ExprNode::Const(v) if !v.is_finite() => return true,
                ExprNode::Unary(OpKind::Recip, a) => {
                    if matches!(self.node(*a), ExprNode::Const(v) if v == 0.0) {
                        return true;
                    }
                    stack.push(*a);
                }
                ExprNode::Binary(OpKind::Div, a, b) => {
                    if matches!(self.node(*b), ExprNode::Const(v) if v == 0.0) {
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
                ExprNode::Nary(_, range) => {
                    let s = range.start as usize;
                    let l = range.len as usize;
                    for child in &self.nary_children[s..s + l] {
                        stack.push(*child);
                    }
                }
                ExprNode::Reduce { body, .. } => stack.push(*body),
                ExprNode::Write { value, .. } => stack.push(*value),
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
            match &self.node(id) {
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
                ExprNode::Nary(_, range) => {
                    let s = range.start as usize;
                    let l = range.len as usize;
                    for child in &self.nary_children[s..s + l] {
                        stack.push(*child);
                    }
                }
                ExprNode::Reduce { body, .. } => stack.push(*body),
                ExprNode::Write { value, .. } => stack.push(*value),
            }
        }
        count
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
        self.splice_with(other, root, |arena, slot| {
            let found = arena.uniform_slot_for(other.uniforms[slot.0 as usize]);
            arena.push_uniform(found)
        })
    }

    /// The walk [`splice`](Self::splice) is: `other`'s fragment copied in,
    /// each uniform slot it reads becoming the node `input` builds for it
    /// here, built once however often the slot is read. `splice` places
    /// each by identity. Buffers merge by identity, as `splice`'s do.
    fn splice_with<F>(&mut self, other: &ExprArena, root: ExprId, mut input: F) -> ExprId
    where
        F: FnMut(&mut ExprArena, UniformId) -> ExprId,
    {
        let mut id_map: Vec<Option<ExprId>> = vec![None; other.len()];
        // Fragment-local BufferId -> this arena's slot, filled lazily.
        let mut buf_map: Vec<Option<BufferId>> = vec![None; other.buffers.len()];
        // Fragment-local UniformId -> the node `input` built for it.
        let mut uni_map: Vec<Option<ExprId>> = vec![None; other.uniforms.len()];

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
                    let new_id = match other.node(id) {
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
                        ExprNode::Uniform(u) => match uni_map[u.0 as usize] {
                            Some(node) => node,
                            None => {
                                let node = input(self, u);
                                uni_map[u.0 as usize] = Some(node);
                                node
                            }
                        },
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
                        ExprNode::Nary(op, range) => {
                            let (s, l) = (range.start as usize, range.len as usize);
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
                        ExprNode::Write {
                            row,
                            col,
                            lane,
                            value,
                        } => {
                            let value = m(value);
                            self.push_write(row, col, lane, value)
                        }
                    };
                    id_map[id.0 as usize] = Some(new_id);
                }
            }
        }

        id_map[root.0 as usize].expect("splice: root was never copied")
    }

    /// This arena's `body`, built against `placeholder`, closed into a fold:
    /// `Reduce(fold_at(b), body[placeholder := Var(b)])`, as an arena of its
    /// own.
    ///
    /// `b` is the lowest [`Binder`] no fold reachable from `body` binds. So
    /// binders are chosen inside-out: a fold sees every inner fold's slot
    /// and takes the next free one, and distinct live binders never share
    /// an index. It is chosen here, after the body exists, because which
    /// slots the body binds decides it — the reason a body is built against
    /// a placeholder at all.
    ///
    /// The one definition of that rule and of the rename: `Kernel::over`
    /// and `kernel!`'s lowering each build a fold through it,
    /// so a fold written in the syntax and the same fold built with the
    /// builder are one program.
    ///
    /// The arena returned holds this arena's buffer and uniform tables, slot
    /// for slot — every declaration, read or not, since a positional binding
    /// supplies them in that order — then the binder's `Var`, then the body,
    /// then the fold, and nothing else: what the placeholder read, and what
    /// renaming it left behind, stay in this arena. The body is copied as a
    /// `Kernel` copies a term, through its DAG, from the root and last child
    /// first: the layout a `Kernel` has always given a fold, so that one
    /// built through here moves no node of its arena (a copy first child
    /// first moved a sum's operands, measured: one term, one key, a
    /// different arena).
    ///
    /// # Errors
    ///
    /// [`IndexSpaceFull`] when the body binds every binder.
    pub fn close_over(
        &self,
        body: ExprId,
        placeholder: Placeholder,
        fold_at: impl FnOnce(Binder) -> Fold,
    ) -> Result<(ExprArena, ExprId), IndexSpaceFull> {
        let binder = self.lowest_free_binder(body).ok_or(IndexSpaceFull)?;
        let fold = fold_at(binder);

        use crate::expr::{ExprBuilderExt, from_arena, substitute_vars, to_arena};
        let (open, tables) = from_arena(self, body);
        let mut closed = Builder::new();
        let index = closed.push_var(binder.var());
        let body = substitute_vars(&mut closed, open.entry(), &[(placeholder.var(), index)]);
        let root = closed.push_reduce(fold, body);
        let closed = closed.finish(&[root]);
        Ok(to_arena(closed.entry(), &tables))
    }

    /// Begin a fold's body in this arena, the `depth`th fold open at once:
    /// this arena becomes a copy of itself, the body is built in it against
    /// the returned fold's [`index`](OpenFold::index), and
    /// [`close_fold`](Self::close_fold) puts the enclosing arena back with
    /// only the closed fold added.
    ///
    /// The two are the one definition of building a fold where its body is
    /// written — `kernel!`'s lowering, at expansion, and an entry that takes
    /// a kernel-typed argument, which runs lowering's steps when it is called
    /// (docs/plans/2026-09-25-the-language-is-kernel.md, Phase D-a) — so a
    /// fold built either way lays out its arena the same way. Every id bound
    /// before the fold opened means the same node in the copy, which is what
    /// lets the body read them; what the placeholder read, and what renaming
    /// it leaves behind, stay in the copy and go with it.
    ///
    /// One placeholder per fold open at once, the `depth`th for `depth`
    /// open, so a nested fold's rename never reaches its enclosing fold's
    /// index. `None` past [`Binder::COUNT`] folds deep: a program cannot
    /// nest more than it has binders.
    #[must_use]
    pub fn open_fold(&mut self, depth: usize) -> Option<OpenFold> {
        let placeholder = Placeholder::nth(depth).filter(|_| depth < Binder::COUNT)?;
        let copy = self.clone();
        let enclosing = core::mem::replace(self, copy);
        let index = self.push_var(placeholder.var());
        Some(OpenFold {
            enclosing,
            placeholder,
            index,
        })
    }

    /// End the fold [`open_fold`](Self::open_fold) began: its `body`, built
    /// in this arena, closed by [`close_over`](Self::close_over) into
    /// `fold_at`'s fold, and spliced into the enclosing arena, which this
    /// arena becomes again. Returns the fold's node there.
    ///
    /// # Errors
    ///
    /// [`IndexSpaceFull`] when the body binds every binder; the enclosing
    /// arena is restored either way.
    pub fn close_fold(
        &mut self,
        open: OpenFold,
        body: ExprId,
        fold_at: impl FnOnce(Binder) -> Fold,
    ) -> Result<ExprId, IndexSpaceFull> {
        let OpenFold {
            enclosing,
            placeholder,
            ..
        } = open;
        let copy = core::mem::replace(self, enclosing);
        let (closed, root) = copy.close_over(body, placeholder, fold_at)?;
        Ok(self.splice(&closed, root))
    }

    /// The fragment at `root` and nothing else: every node `root` reaches,
    /// in this arena's order, and every buffer and uniform this arena
    /// declares, slot for slot — read or not, since a positional binding
    /// supplies them in that order. What building left behind (a node a
    /// substitution replaced, a `let` nothing read) is dropped.
    ///
    /// It matters because some passes ask questions of a *whole* arena — a
    /// fast path that skips a rebuild when no node needs one — and a rebuild
    /// can move nodes, so an arena carrying construction garbage could
    /// compile to other bytes than the same program without it, under one
    /// cache key.
    #[must_use]
    pub fn compact(&self, root: ExprId) -> (ExprArena, ExprId) {
        self.relink(root, &self.buffers, &self.uniforms)
    }

    /// The lowest binder no `Reduce` reachable from `body` binds, or `None`
    /// when every one is bound: [`ExprArena::close_over`]'s choice.
    fn lowest_free_binder(&self, body: ExprId) -> Option<Binder> {
        let mut bound = [false; Binder::COUNT];
        let mut seen = vec![false; self.len()];
        let mut stack = vec![body];
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            // `ExprNode::Reduce`'s `Fold` is why this is one line. Read off
            // a `Const` child it was a float, tested against `floorf` and a
            // magic range, and asked again by every pass that wanted a
            // binder.
            if let ExprNode::Reduce { fold, .. } = self.node(id) {
                bound[usize::from(fold.binder().slot())] = true;
            }
            stack.extend(self.children(id));
        }
        Binder::all().find(|binder| !bound[usize::from(binder.slot())])
    }

    /// The field at `root` observed at `(u, v)`: contramap,
    /// `⟦warp(f, u, v)⟧(x, y) = ⟦f⟧(⟦u⟧(x, y), ⟦v⟧(x, y))`. `u` and `v` are
    /// fields of the outer coordinates, nodes of this arena, and the two
    /// substitutions are simultaneous: `(X − Y)` at `(Y, X)` is `Y − X`.
    ///
    /// The one arena-level definition. `kernel!`'s `.at` lowers to it at
    /// both of its sites, and a kernel-typed argument's application is a
    /// splice and then this ([`ExprArena::apply`]); `Kernel::at` is the same
    /// substitution over a `Kernel`'s DAG, and `kernel!`'s tests pin the two
    /// to one program.
    ///
    /// **A derivative is of the warped field.** `Dwrt` is left for the
    /// runtime tier to resolve, so the substitution reaches its operand:
    /// `DX(f)` at `(u, v)` is `∂/∂X (f ∘ (u, v))`, the chain rule, not
    /// `(∂f/∂X) ∘ (u, v)`. That is deliberate — it is what keeps a glyph's
    /// antialiasing ramp one *screen* pixel wide at any scale — and the two
    /// readings agree wherever the warp is a translation, its Jacobian being
    /// the identity (`pixelflow-compiler/tests/derivative_under_warp.rs`).
    ///
    /// **Fold binders are not substituted, and cannot be captured.** Only
    /// the coordinate axes are rewritten. A binder free in `u` or `v` is a
    /// fold's placeholder while that fold's body is built, outside every
    /// slot an inner fold of `f` can hold, and the slot it is renamed to on
    /// closing is the lowest one nothing in the body binds — `f`'s folds
    /// included ([`ExprArena::close_over`]).
    ///
    /// **A name is expanded first.** A substitution cannot reach through a
    /// [`ExprNode::Ref`] — it has no `Var` to rewrite here, only a key — so
    /// left in place it would sample its referent at the *outer*
    /// coordinates: plausible pixels, wrong ones. At `(X, Y)` there is
    /// nothing to substitute and `root` is returned as it stands, a name
    /// still a name and still its own optimization unit.
    pub fn warp(&mut self, root: ExprId, [u, v]: [ExprId; 2]) -> ExprId {
        let at_the_sample = self.node(u) == ExprNode::Var(Axis::X.var())
            && self.node(v) == ExprNode::Var(Axis::Y.var());
        if at_the_sample {
            return root;
        }
        let linked = crate::passes::expand_refs(self, root);
        self.substitute_vars_with(linked, &[(Axis::X.var(), u), (Axis::Y.var(), v)])
    }

    /// Rebuild the subgraph at `root`, replacing every `Var(i)` for which
    /// `subs` has an entry with the given (already existing) node — the
    /// generic substitution [`warp`](Self::warp) is built on, which is what
    /// `Kernel::at` is built from.
    ///
    /// Entries must reference nodes already in this arena (e.g. from
    /// [`ExprArena::splice`]). Unlisted variables are preserved. Returns the
    /// new root in the same arena; old nodes become unreachable garbage,
    /// which is fine for an append-only arena.
    pub fn substitute_vars_with(&mut self, root: ExprId, subs: &[(u8, ExprId)]) -> ExprId {
        let lookup = |i: u8| subs.iter().find(|(v, _)| *v == i).map(|(_, id)| *id);

        let old_len = self.len();
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
                    let new_id = match self.node(id) {
                        ExprNode::Var(i) => match lookup(i) {
                            Some(replacement) => replacement,
                            None => self.push_var(i),
                        },
                        ExprNode::Const(v) => self.push_const(v),
                        ExprNode::Param(i) => self.push_param(i),
                        ExprNode::Buffer(b) => self.push_buffer(b),
                        ExprNode::Uniform(u) => self.push_uniform(u),
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
                        ExprNode::Nary(op, range) => {
                            let (s, l) = (range.start as usize, range.len as usize);
                            let child_ids: Vec<ExprId> = self.nary_children[s..s + l].to_vec();
                            let mapped: Vec<ExprId> = child_ids.into_iter().map(m).collect();
                            self.push_nary(op, &mapped)
                        }
                        ExprNode::Reduce { fold, body } => {
                            let body = m(body);
                            self.push_reduce(fold, body)
                        }
                        ExprNode::Write {
                            row,
                            col,
                            lane,
                            value,
                        } => {
                            let value = m(value);
                            self.push_write(row, col, lane, value)
                        }
                    };
                    id_map[id.0 as usize] = Some(new_id);
                }
            }
        }

        id_map[root.0 as usize].expect("substitute_vars_with: root was never rebuilt")
    }

    // ───────────────────── linking ───────────────────────────

    /// This arena with its buffer and uniform tables replaced slot for slot:
    /// slot `i` names `buffers[i]` / `uniforms[i]`, and no node moves.
    ///
    /// The second half of sharing one optimization between two compositions
    /// of one shape (`pixelflow-search`'s runtime cache): the saturated graph
    /// carries the first composition's names in its leaves, and this gives
    /// the extracted term the second's. Positional, so the tables must
    /// already agree on everything but the names — the slot count, and a
    /// buffer's extents, which the code folds its addressing against. A
    /// uniform's default is the block's business, not the code's, and may
    /// differ.
    ///
    /// # Panics
    ///
    /// Panics if a table's length differs from this arena's, or a buffer's
    /// extents differ from the one whose slot it takes.
    #[must_use]
    pub fn with_tables(mut self, buffers: Vec<BufferDecl>, uniforms: Vec<UniformDecl>) -> Self {
        assert_eq!(
            self.buffers.len(),
            buffers.len(),
            "with_tables: the buffer tables differ in length"
        );
        for (slot, (mine, theirs)) in self.buffers.iter().zip(&buffers).enumerate() {
            assert!(
                mine.width == theirs.width && mine.height == theirs.height,
                "with_tables: buffer slot {slot} changes extents: {mine:?} -> {theirs:?}"
            );
        }
        assert_eq!(
            self.uniforms.len(),
            uniforms.len(),
            "with_tables: the uniform tables differ in length"
        );
        self.buffers = buffers;
        self.uniforms = uniforms;
        self
    }

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
        let mut reachable = vec![false; self.len()];
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
        // The link by identity, built once: a search of it per occurrence
        // was quadratic in the arguments, and nothing bounds those. First
        // entry wins, as `position` did, should an order name one twice.
        let mut uniform_index: Memo<UniformIdentity, u64> = Memo::new();
        for (i, decl) in uniforms.iter().enumerate() {
            uniform_index.entry(decl.id).or_insert(i as u64);
        }
        let uniform_slot = |u: UniformId| -> UniformId {
            let decl = self.uniforms[u.0 as usize];
            let i = *uniform_index
                .get(&decl.id)
                .unwrap_or_else(|| panic!("relink: reachable {decl:?} is not in the link"));
            assert_eq!(
                uniforms[i as usize], decl,
                "relink: uniform declaration disagrees"
            );
            UniformId(i)
        };

        let mut out = ExprArena::with_capacity(self.len());
        out.buffers = buffers.to_vec();
        out.uniforms = uniforms.to_vec();
        let mut dense: Vec<Option<ExprId>> = vec![None; self.len()];
        for idx in 0..self.len() {
            if !reachable[idx] {
                continue;
            }
            let id = ExprId(idx as u32);
            let node = self.node(id);
            let m =
                |old: ExprId| dense[old.0 as usize].expect("relink: child densified before parent");
            let new_id = match &node {
                ExprNode::Var(i) => out.push_var(*i),
                ExprNode::Const(v) => out.push_const(*v),
                ExprNode::Param(i) => out.push_param(*i),
                ExprNode::Buffer(b) => out.push_buffer(buffer_slot(*b)),
                ExprNode::Uniform(u) => out.push_uniform(uniform_slot(*u)),
                ExprNode::Ref(k) => out.push_ref(*k),
                ExprNode::Unary(op, a) => out.push_unary(*op, m(*a)),
                ExprNode::Binary(op, a, b) => out.push_binary(*op, m(*a), m(*b)),
                ExprNode::Ternary(op, a, b, c) => out.push_ternary(*op, m(*a), m(*b), m(*c)),
                ExprNode::Nary(op, range) => {
                    let (s, l) = (range.start as usize, range.len as usize);
                    let mapped: Vec<ExprId> =
                        self.nary_children[s..s + l].iter().map(|c| m(*c)).collect();
                    out.push_nary(*op, &mapped)
                }
                ExprNode::Reduce { fold, body } => {
                    let body = m(*body);
                    out.push_reduce(*fold, body)
                }
                ExprNode::Write {
                    row,
                    col,
                    lane,
                    value,
                } => {
                    let value = m(*value);
                    out.push_write(*row, *col, *lane, value)
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
                Task::Visit(id) => match &self.node(id) {
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
                    ExprNode::Nary(op, range) => {
                        let s = range.start as usize;
                        let l = range.len as usize;
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
                        write!(f, "{}[{fold}](", OpKind::Reduce.name())?;
                    }
                    ExprNode::Write {
                        row,
                        col,
                        lane,
                        value,
                    } => {
                        stack.push(Task::WriteStr(")"));
                        stack.push(Task::Visit(*value));
                        write!(
                            f,
                            "Write[row=i{}, col=i{}, lane=i{}](",
                            row.var(),
                            col.var(),
                            lane.var()
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
            let s_node = self.node(s_id);
            let o_node = other.node(o_id);

            match (&s_node, &o_node) {
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
                (ExprNode::Nary(s_op, s_range), ExprNode::Nary(o_op, o_range)) => {
                    if s_op != o_op || s_range.len != o_range.len {
                        return false;
                    }
                    let ss = s_range.start as usize;
                    let os = o_range.start as usize;
                    let len = s_range.len as usize;
                    for i in 0..len {
                        stack.push((self.nary_children[ss + i], other.nary_children[os + i]));
                    }
                }
                (
                    ExprNode::Write {
                        row: s_row,
                        col: s_col,
                        lane: s_lane,
                        value: s_value,
                    },
                    ExprNode::Write {
                        row: o_row,
                        col: o_col,
                        lane: o_lane,
                        value: o_value,
                    },
                ) => {
                    if s_row != o_row || s_col != o_col || s_lane != o_lane {
                        return false;
                    }
                    stack.push((*s_value, *o_value));
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

    // ───────────────────────── close_over ─────────────────────────

    /// A body `X·p + Y` over the `n`th placeholder `p`, and the arena it is
    /// built in.
    fn open_body(n: usize) -> (ExprArena, ExprId, Placeholder) {
        let placeholder = Placeholder::nth(n).expect("a placeholder");
        let mut a = ExprArena::new();
        let (x, y) = (a.push_var(Axis::X.var()), a.push_var(Axis::Y.var()));
        let p = a.push_var(placeholder.var());
        let xp = a.push_binary(OpKind::Mul, x, p);
        let body = a.push_binary(OpKind::Add, xp, y);
        (a, body, placeholder)
    }

    /// Whether a `Var` at or past the first placeholder is anywhere in `a`.
    fn holds_a_placeholder(a: &ExprArena) -> bool {
        let first = Placeholder::nth(0).expect("a placeholder").var();
        a.nodes()
            .any(|(_, node)| matches!(node, ExprNode::Var(v) if v >= first))
    }

    /// `close_over` binds the lowest slot the body leaves free, renames the
    /// placeholder to it, and hands back only the fold: here, over a body
    /// already holding a fold at slot 0, slot 1.
    #[test]
    fn close_over_binds_the_lowest_free_binder_and_renames_the_placeholder() {
        let placeholder = Placeholder::nth(0).expect("a placeholder");
        let mut a = ExprArena::new();
        let x = a.push_var(Axis::X.var());
        let slot0 = Binder::from_slot(0).expect("slot 0");
        let inner = a.push_reduce(Fold::new(Monoid::SUM, slot0, 0..3), x);
        let p = a.push_var(placeholder.var());
        let body = a.push_binary(OpKind::Mul, inner, p);

        let (closed, root) = a
            .close_over(body, placeholder, |binder| {
                Fold::new(Monoid::SUM, binder, 0..4)
            })
            .expect("a free binder");
        let ExprNode::Reduce { fold, body } = closed.node(root) else {
            panic!("a fold, got {}", closed.display(root));
        };
        assert_eq!(fold.binder().slot(), 1, "slot 0 is bound inside");
        let ExprNode::Binary(OpKind::Mul, _, index) = closed.node(body) else {
            panic!("the product, got {}", closed.display(body));
        };
        assert_eq!(closed.node(index), ExprNode::Var(fold.binder().var()));
        assert!(!holds_a_placeholder(&closed), "{}", closed.display(root));
    }

    /// The closed arena keeps every declaration of the one the body was
    /// built in, read or not and in order, since a positional binding
    /// supplies them so.
    #[test]
    fn close_over_keeps_every_declaration_in_order() {
        let (mut a, body, placeholder) = open_body(3);
        let unread = UniformDecl {
            id: UniformIdentity::mint(),
            default: 1.0,
        };
        a.declare_uniform(unread);
        let (closed, _) = a
            .close_over(body, placeholder, |binder| {
                Fold::new(Monoid::MAX, binder, 0..2)
            })
            .expect("a free binder");
        assert_eq!(closed.uniforms(), [unread]);
    }

    /// A body binding every slot has none left: the fold would be one
    /// deeper than the index space.
    #[test]
    fn close_over_refuses_a_body_that_binds_every_binder() {
        let (mut a, mut body, placeholder) = open_body(0);
        for binder in Binder::all() {
            body = a.push_reduce(Fold::new(Monoid::SUM, binder, 0..1), body);
        }
        let unclosed = a.close_over(body, placeholder, |binder| {
            Fold::new(Monoid::SUM, binder, 0..1)
        });
        assert_eq!(unclosed.err(), Some(IndexSpaceFull));
    }

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
        assert!(matches!(arena.node(children[0]), ExprNode::Buffer(b) if b == buf));
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
        assert!(matches!(arena.node(red), ExprNode::Reduce { fold: f, .. } if f == fold));
    }

    /// A store names its binders and holds its value: one child, three
    /// indices that are metadata, and an identity that tells two stores of
    /// one value at different lattice positions apart.
    #[test]
    fn a_write_stores_one_value_under_three_binders() {
        let slot = |s: u8| Binder::from_slot(s).expect("a binder slot");
        let (row, col, lane) = (slot(0), slot(1), slot(2));
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let l = arena.push_var(lane.var());
        let value = arena.push_binary(OpKind::Add, x, l);
        let write = arena.push_write(row, col, lane, value);

        let children: Vec<ExprId> = arena.children(write).collect();
        assert_eq!(children, vec![value], "the value is the one child");
        assert!(matches!(
            arena.node(write),
            ExprNode::Write { row: r, col: c, lane: n, value: v }
                if r == row && c == col && n == lane && v == value
        ));
        assert_eq!(
            format!("{}", arena.display(write)),
            "Write[row=i4, col=i5, lane=i6](add(Var(0), Var(6)))"
        );

        // The same value stored under other binders is a different store.
        let elsewhere = arena.push_write(col, row, lane, value);
        assert!(arena.subtree_eq(write, &arena, write));
        assert!(!arena.subtree_eq(write, &arena, elsewhere));
    }

    /// A store is an effect, not an operation: it has no `OpKind` to price
    /// or rewrite, and asking for one is a pipeline that reached a `Write`
    /// where only values belong.
    #[test]
    #[should_panic(expected = "a Write is a store, not an operation")]
    fn a_write_has_no_kind() {
        let slot = |s: u8| Binder::from_slot(s).expect("a binder slot");
        let mut arena = ExprArena::new();
        let value = arena.push_var(0);
        let write = arena.push_write(slot(0), slot(1), slot(2), value);
        let kind = arena.kind(write);
        unreachable!("a Write answered with {kind:?}");
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

    /// A node stays small enough that nobody boxed anything by accident.
    ///
    /// The bound is a tripwire and not a budget — see the compile-time
    /// assertion's comment. It is deliberately loose, so a new variant is a
    /// question about what the language means rather than about bytes.
    #[test]
    fn size_of_expr_node() {
        assert!(
            core::mem::size_of::<ExprNode>() <= 32,
            "ExprNode is {} bytes, expected <= 32",
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

    // ───────────────────────── free_index ─────────────────────────

    /// An index is free where no fold around the read binds it: a
    /// placeholder always, a binder outside its `Reduce`, and neither a
    /// coordinate nor a binder under the `Reduce` that binds it.
    #[test]
    fn an_index_is_free_only_outside_the_fold_that_binds_it() {
        let mut a = ExprArena::new();
        let x = a.push_var(Axis::X.var());
        assert_eq!(a.free_index(x), None, "a coordinate is read, never bound");

        let binder = Binder::from_slot(1).expect("a binder");
        let i = a.push_var(binder.var());
        assert_eq!(a.free_index(i), Some(binder.var()));
        let body = a.push_binary(OpKind::Mul, x, i);
        let fold = a.push_reduce(Fold::new(Monoid::SUM, binder, 0..4), body);
        assert_eq!(a.free_index(fold), None, "bound under its fold");

        // The same binder read beside the fold, outside it, is free: a
        // scan for "some binder Var is reachable" cannot tell this from the
        // line above.
        let beside = a.push_binary(OpKind::Add, fold, i);
        assert_eq!(a.free_index(beside), Some(binder.var()));

        let placeholder = Placeholder::nth(0).expect("a placeholder");
        let p = a.push_var(placeholder.var());
        let both = a.push_binary(OpKind::Add, beside, p);
        assert_eq!(
            a.free_index(both),
            Some(binder.var()),
            "the lowest free index, the binder below the placeholder"
        );
        let open = a.push_binary(OpKind::Add, fold, p);
        assert_eq!(a.free_index(open), Some(placeholder.var()));
    }

    // ───────────────────── open_fold / close_fold ─────────────────────

    /// A fold built where its body is written is `close_over`'s fold of
    /// that body, spliced into the enclosing arena — and the enclosing arena
    /// gains that and nothing else: the placeholder stayed in the copy.
    #[test]
    fn a_fold_opened_and_closed_is_close_overs_and_leaves_nothing_behind() {
        let mut a = ExprArena::new();
        let x = a.push_var(Axis::X.var());
        let before = a.len();
        let open = a.open_fold(0).expect("depth 0 is in range");
        let index = open.index();
        let body = a.push_binary(OpKind::Mul, x, index);
        let fold = a
            .close_fold(open, body, |b| Fold::new(Monoid::SUM, b, 0..4))
            .expect("one fold binds one binder");

        let ExprNode::Reduce { fold: range, body } = a.node(fold) else {
            panic!("a fold, got {}", a.display(fold));
        };
        assert_eq!(range.binder().slot(), 0);
        assert_eq!(
            a.node(body),
            ExprNode::Binary(OpKind::Mul, x, a.push_var(range.binder().var()))
        );
        assert_eq!(a.free_index(fold), None);
        assert_eq!(
            a.len(),
            before + 3,
            "the binder's Var, the body and the fold; no placeholder"
        );
    }

    /// No deeper than the index space: one fold per binder.
    #[test]
    fn a_fold_opens_no_deeper_than_the_binders() {
        let mut a = ExprArena::new();
        assert!(a.open_fold(Binder::COUNT - 1).is_some());
        assert!(a.open_fold(Binder::COUNT).is_none());
    }

    // ───────────────────────── compact ─────────────────────────

    /// What `root` does not reach is dropped, and every declaration is kept
    /// in its slot, read or not.
    #[test]
    fn compact_keeps_the_program_and_every_declaration() {
        let mut a = ExprArena::new();
        let unread = a.declare_uniform(UniformDecl {
            id: UniformIdentity::mint(),
            default: 1.0,
        });
        let read = a.declare_uniform(UniformDecl {
            id: UniformIdentity::mint(),
            default: 2.0,
        });
        let _garbage = a.push_uniform(unread);
        let x = a.push_var(Axis::X.var());
        let _also_garbage = a.push_binary(OpKind::Sub, x, x);
        let u = a.push_uniform(read);
        let root = a.push_binary(OpKind::Add, x, u);

        let (compact, compact_root) = a.compact(root);
        assert_eq!(compact.len(), 3, "X, the read uniform, the sum");
        assert_eq!(compact.uniforms(), a.uniforms(), "every slot, in order");
        assert!(compact.subtree_eq(compact_root, &a, root));
    }
}

#[cfg(test)]
mod composition_tests {
    use super::*;

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
    fn subtree_eq_distinguishes_uniform_declarations() {
        let (a, ra) = uniform_fragment(uniform_decl(1.0));
        let (b, rb) = uniform_fragment(uniform_decl(1.0));
        assert!(a.subtree_eq(ra, &a, ra));
        assert!(!a.subtree_eq(ra, &b, rb), "same slot, different instance");
    }

    /// A fragment spliced with its uniforms placed by the caller is one
    /// application of a function of them: `u·s + X` over an input `u` and
    /// a shared term `s`, applied at two of the host's uniforms, is the two
    /// terms written in the host by hand — each input built once however
    /// often it is read, the shared term one node, and nothing declared.
    #[test]
    fn splicing_with_placed_uniforms_applies_the_fragment() {
        let mut template = ExprArena::new();
        let [u, s] = [0, 1].map(|_| template.declare_uniform(uniform_decl(f32::NAN)));
        let (u, s) = (template.push_uniform(u), template.push_uniform(s));
        let x = template.push_var(0);
        let us = template.push_binary(OpKind::Mul, u, s);
        let uu = template.push_binary(OpKind::Mul, u, u);
        let body = template.push_binary(OpKind::Add, us, x);
        let body = template.push_binary(OpKind::Sub, body, uu);

        let mut host = ExprArena::new();
        let slots = [1.0, 2.0].map(|v| host.declare_uniform(uniform_decl(v)));
        let x = host.push_var(0);
        let shared = host.push_unary(OpKind::Sqrt, x);
        let mut built = 0;
        let copies = slots.map(|slot| {
            host.splice_with(&template, body, |arena, input| {
                built += 1;
                match input.0 {
                    0 => arena.push_uniform(slot),
                    _ => shared,
                }
            })
        });
        assert_eq!(
            built, 4,
            "each input built once per copy, though `u` is read thrice"
        );
        assert_eq!(host.uniforms().len(), 2, "nothing declared");

        let by_hand = slots.map(|slot| {
            let u = host.push_uniform(slot);
            let us = host.push_binary(OpKind::Mul, u, shared);
            let uu = host.push_binary(OpKind::Mul, u, u);
            let term = host.push_binary(OpKind::Add, us, x);
            host.push_binary(OpKind::Sub, term, uu)
        });
        assert_eq!(copies, by_hand, "hash-consed onto the same nodes");
        assert_ne!(copies[0], copies[1]);
    }
}
