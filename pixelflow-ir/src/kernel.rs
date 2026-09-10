//! `Kernel` — the language's runtime value.
//!
//! A `Kernel` is a handle to an expression fragment: an [`ExprArena`] plus its
//! root. It is the value the front end (the `kernel!` macro) produces and the
//! thing consumers compose — `sum`, `at`, `select`, arithmetic — with the
//! arena hidden entirely behind the methods. This is the "JIT-first" surface:
//! programs are built as `Kernel` values (our own AST), type-checked and
//! monomorphized by our codegen at `compile` time, never encoded in Rust's
//! type system.
//!
//! Composition is arena splicing: every method clones the receiver's arena,
//! splices the operands in (DAG-preserving), and appends the new node. Values
//! are immutable and cheaply cloned (`Arc`); the deep copy happens only when a
//! new node is built, which is construction/bake time, not per pixel.

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::arena::{BufferDecl, BufferIdentity, ExprArena, ExprId, UniformDecl, UniformIdentity};
use crate::dag::{Builder, Dag, Node, Rooted};
use crate::expr::{Environment, ExprBuilderExt, ExprData, copy_subgraph, splice, substitute_vars};
use crate::fold::{Binder, Fold, Monoid};
use crate::kind::OpKind;

/// One bit per placeholder index, set while that index is claimed by a binder
/// under construction. Claims are taken and released in any order, so this is a
/// set — not a depth counter. A counter would be correct only if every release
/// were the most recent claim, which holds within one thread's nesting and
/// fails the moment two threads build binders at once: thread A releases ticket
/// 0 while B still holds 1, and the next claim hands out 1 again.
static PLACEHOLDERS_IN_USE: AtomicU64 = AtomicU64::new(0);

/// Placeholder indices sit above the retired coordinate space (`0..4`, of
/// which only X and Y are live) and the reduction index space (`4..8`). A `Kernel` never contains the compiler's manifold-param
/// slots (the value-producing macro path rejects manifold params outright), so
/// everything from 8 up is free.
const PLACEHOLDER_BASE: u32 = 8;

/// A reduction's bound index while its body is under construction, before a
/// real slot (`4..8`) is chosen.
///
/// The placeholder must be unique among binders that are *simultaneously* being
/// built: a nested fold renames every occurrence of its own placeholder to a
/// real slot, so if it shared one with the fold enclosing it, it would capture
/// the outer index — `Σ_i Σ_j f(i, j)` would silently become `Σ_i Σ_j f(j, j)`.
/// The claim is released on drop, so the space is bounded by how many binders
/// are open at this instant, not by how many kernels have ever been built.
///
/// [`lowest_free_binder`] caps nesting at [`Binder::COUNT`], so the 64 placeholders here
/// admit 16 fully-nested concurrent constructions; exhaustion panics rather
/// than aliasing an index.
struct BinderScope(u32);

impl BinderScope {
    fn enter() -> Self {
        let mut in_use = PLACEHOLDERS_IN_USE.load(Ordering::Relaxed);
        loop {
            let bit = (!in_use).trailing_zeros();
            assert!(
                bit < u64::BITS,
                "too many kernel binders under construction at once"
            );
            match PLACEHOLDERS_IN_USE.compare_exchange_weak(
                in_use,
                in_use | (1 << bit),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Self(bit),
                Err(observed) => in_use = observed,
            }
        }
    }

    fn placeholder(&self) -> u8 {
        (PLACEHOLDER_BASE + self.0) as u8
    }
}

impl Drop for BinderScope {
    fn drop(&mut self) {
        PLACEHOLDERS_IN_USE.fetch_and(!(1 << self.0), Ordering::Relaxed);
    }
}

/// The lowest binder not already bound by a `Reduce` in `arena`.
///
/// Binders are built inside-out, so a fold sees every inner fold's slot and
/// takes the next free one — distinct live binders never share an index.
///
/// # Panics
///
/// Panics when every slot is live, i.e. one fold deeper than the index space.
fn lowest_free_binder(dag: &Dag<ExprData>) -> Binder {
    let mut used = [false; Binder::COUNT];
    for node in dag.iter() {
        // `ExprData::Reduce(Fold)` is why this is two lines. Read off a
        // `Const` child it was a float, tested against `floorf` and a magic
        // range, and asked again by every pass that wanted a binder.
        if let ExprData::Reduce(fold) = *node {
            used[fold.binder().slot() as usize] = true;
        }
    }
    Binder::all()
        .find(|b| !used[b.slot() as usize])
        .unwrap_or_else(|| {
            panic!(
                "more than {} live nested reductions: the index space is full",
                Binder::COUNT
            )
        })
}

/// A named scalar argument of a kernel: the JIT tier's spelling of a
/// builder's struct field.
///
/// Creating one mints an identity; the handle is the only way to set the
/// value later, so a kernel's arguments are exactly the handles its author
/// kept. It composes as a leaf ([`Uniform::kernel`]) or stands in for a
/// builder's scalar parameter ([`Scalar`]); either way the value is invariant
/// across the lattice and unknown until the call, so the compiler hoists
/// everything that depends only on it into the per-call prologue and never
/// folds it.
///
/// Two handles from two `new` calls are two arguments, even with equal
/// defaults; one handle read from twenty places is one argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Uniform {
    decl: UniformDecl,
}

impl Uniform {
    /// A new argument, holding `default` until a block binds it.
    #[must_use]
    pub fn new(default: f32) -> Self {
        Self {
            decl: UniformDecl {
                id: UniformIdentity::mint(),
                default,
            },
        }
    }

    /// The declaration this handle carries into every arena that reads it.
    #[must_use]
    pub fn decl(self) -> UniformDecl {
        self.decl
    }

    /// Which argument this is.
    #[must_use]
    pub fn identity(self) -> UniformIdentity {
        self.decl.id
    }

    /// The value the kernel holds for this argument when nothing binds it.
    #[must_use]
    pub fn default_value(self) -> f32 {
        self.decl.default
    }

    /// The leaf, as a fragment: composes like any [`Kernel`].
    #[must_use]
    pub fn kernel(self) -> Kernel {
        let mut b = Builder::new();
        let mut env = Environment::new();
        let slot = env.slot_for_uniform(self.decl);
        let r = b.push_uniform(slot);
        Kernel::wrap(b.finish(&[r]), env, BTreeMap::new())
    }
}

/// What a builder accepts for a scalar parameter. The *type* decides whether
/// the value is folded into the fragment as a constant or declared as a
/// uniform slot: an `f32` folds, so every call site that passes one keeps its
/// meaning, and a [`Uniform`] handle makes the parameter an argument of the
/// compiled kernel instead.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    /// Folded in: part of the kernel.
    Const(f32),
    /// Bound per call: an argument of the kernel.
    Uniform(Uniform),
}

impl From<f32> for Scalar {
    fn from(v: f32) -> Self {
        Self::Const(v)
    }
}

impl From<Uniform> for Scalar {
    fn from(u: Uniform) -> Self {
        Self::Uniform(u)
    }
}

/// A composed expression fragment: the front-end value.
#[derive(Clone)]
pub struct Kernel {
    inner: Arc<KernelData>,
}

struct KernelData {
    rooted: Rooted<ExprData>,
    env: Environment,
    legacy: (ExprArena, ExprId),
    /// Tabulations this kernel carries, by the [`BufferIdentity`] each was
    /// seeded under — the data travelling with the value, so a consumer
    /// never carries a binding beside the kernel that needs it
    /// (docs/plans/2026-09-09-composition-is-linking.md §6). `Arc<[f32]>`,
    /// not `Arc<Vec<f32>>`: the tabulation is immutable once seeded, so the
    /// `Vec` header is a wasted indirection, and every combinator below
    /// merges two of these by cloning the `Arc` — a refcount bump, never a
    /// deep copy.
    buffers: BTreeMap<BufferIdentity, Arc<[f32]>>,
}

/// Merge `other`'s tabulations into `base`. An identity new to `base` is
/// inserted (a refcount clone of the `Arc`, never the data); an identity
/// already present must name the very same tabulation — two kernels
/// composed under one `BufferIdentity` are, by construction
/// (`BufferIdentity::mint`'s doc), reads of the very same buffer, so a
/// pointer that disagrees is a programming error to fail loudly on, not a
/// pair of tabulations to silently pick between.
fn merge_buffer_data(
    base: &mut BTreeMap<BufferIdentity, Arc<[f32]>>,
    other: &BTreeMap<BufferIdentity, Arc<[f32]>>,
) {
    for (id, data) in other {
        match base.get(id) {
            Some(existing) => assert!(
                Arc::ptr_eq(existing, data),
                "Kernel: {id:?} names two different tabulations"
            ),
            None => {
                base.insert(*id, Arc::clone(data));
            }
        }
    }
}

impl Kernel {
    fn wrap(
        rooted: Rooted<ExprData>,
        env: Environment,
        buffers: BTreeMap<BufferIdentity, Arc<[f32]>>,
    ) -> Self {
        let legacy = rooted.entry().marshal(&env);
        Self {
            inner: Arc::new(KernelData {
                rooted,
                env,
                legacy,
                buffers,
            }),
        }
    }

    /// Adopt an already-built fragment.
    #[must_use]
    pub fn from_rooted(
        rooted: Rooted<ExprData>,
        buffers: Vec<BufferDecl>,
        uniforms: Vec<UniformDecl>,
    ) -> Self {
        let root = rooted.entry();
        assert!(
            root.retired_axis().is_none(),
            "Kernel::from_rooted: the expression names Var({}), which was the {} coordinate; a lattice has {} axes and a per-call scalar is a Uniform",
            root.retired_axis().unwrap_or_default(),
            if root.retired_axis() == Some(2) {
                "Z"
            } else {
                "W"
            },
            crate::arena::COORD_AXES,
        );
        let env = Environment { buffers, uniforms };
        Self::wrap(rooted, env, BTreeMap::new())
    }

    /// The root expression node handle.
    #[must_use]
    pub fn root(&self) -> Node<'_, ExprData> {
        self.inner.rooted.entry()
    }

    /// The DAG structure.
    #[must_use]
    pub fn dag(&self) -> &Dag<ExprData> {
        &self.inner.rooted
    }

    /// The rooted DAG.
    #[must_use]
    pub fn rooted(&self) -> &Rooted<ExprData> {
        &self.inner.rooted
    }

    /// Buffer declarations.
    #[must_use]
    pub fn buffers(&self) -> &[BufferDecl] {
        &self.inner.env.buffers
    }

    /// Uniform declarations.
    #[must_use]
    pub fn uniforms(&self) -> &[UniformDecl] {
        &self.inner.env.uniforms
    }

    // ─────────────────────────── leaves ───────────────────────────

    /// The X coordinate.
    #[must_use]
    pub fn x() -> Self {
        Self::coord(0)
    }
    /// The Y coordinate.
    #[must_use]
    pub fn y() -> Self {
        Self::coord(1)
    }
    fn coord(i: u8) -> Self {
        let mut b = Builder::new();
        let r = b.push_var(i);
        Self::wrap(b.finish(&[r]), Environment::new(), BTreeMap::new())
    }

    /// A constant.
    #[must_use]
    pub fn constant(v: f32) -> Self {
        let mut b = Builder::new();
        let r = b.push_const(v);
        Self::wrap(b.finish(&[r]), Environment::new(), BTreeMap::new())
    }

    /// Adopt an already-built fragment — the `kernel!` macro's entry point.
    ///
    /// # Panics
    ///
    /// Panics if the arena names a retired coordinate axis (`Var(2)` or
    /// `Var(3)`, the old Z and W). A lattice has
    /// [`COORD_AXES`](crate::arena::COORD_AXES) axes; a scalar that is the
    /// same at every sample is a [`Uniform`], not an axis of extent 1. This
    /// is where the refusal lives because `Var` is also a reduction binder's
    /// index and a rewrite rule's metavariable, and a `Kernel` is the one
    /// thing that becomes machine code.
    #[must_use]
    pub fn from_parts(arena: ExprArena, root: ExprId) -> Self {
        let (rooted, env) = Rooted::unmarshal(&arena, &[root]);
        let entry = rooted.entry();
        assert!(
            entry.retired_axis().is_none(),
            "Kernel::from_parts: the arena names Var({}), which was the {} \
             coordinate; a lattice has {} axes and a per-call scalar is a \
             Uniform (docs/plans/2026-09-06-lattice-is-the-index.md)",
            entry.retired_axis().unwrap_or_default(),
            if entry.retired_axis() == Some(2) {
                "Z"
            } else {
                "W"
            },
            crate::arena::COORD_AXES,
        );
        Self {
            inner: Arc::new(KernelData {
                rooted,
                env,
                legacy: (arena, root),
                buffers: BTreeMap::new(),
            }),
        }
    }

    // ───────────────────── the builder seam ───────────────────────
    //
    // Every splice below is one of these four methods (`combine`,
    // `combine3`, `sum`, `at`), so they are the chokepoint: each merges the
    // operands' buffer tables (`merge_buffer_data`) alongside the arena
    // splice it already did. `map`/`dwrt`/`Bits::shl` touch only `self`'s
    // arena, so they carry `self`'s table forward unchanged, and `over`
    // carries its `body`'s.

    /// Apply a unary node.
    fn map(&self, op: OpKind) -> Self {
        let mut b = Builder::new();
        let r = copy_subgraph(&mut b, self.root());
        let root = b.push_unary(op, r);
        Self::wrap(
            b.finish(&[root]),
            self.inner.env.clone(),
            self.inner.buffers.clone(),
        )
    }

    /// Apply a binary node with `self` on the left and `rhs` spliced in.
    fn combine(&self, rhs: &Kernel, op: OpKind) -> Self {
        let mut b = Builder::new();
        let mut env = self.inner.env.clone();
        let lhs_root = copy_subgraph(&mut b, self.root());
        let rhs_root = splice(&mut b, &mut env, rhs.root(), &rhs.inner.env);
        let root = b.push_binary(op, lhs_root, rhs_root);
        let mut buffers = self.inner.buffers.clone();
        merge_buffer_data(&mut buffers, &rhs.inner.buffers);
        Self::wrap(b.finish(&[root]), env, buffers)
    }

    /// Apply a ternary node with `self` first and `b`, `c` spliced in.
    fn combine3(&self, b: &Kernel, c: &Kernel, op: OpKind) -> Self {
        let mut builder = Builder::new();
        let mut env = self.inner.env.clone();
        let a_root = copy_subgraph(&mut builder, self.root());
        let b_root = splice(&mut builder, &mut env, b.root(), &b.inner.env);
        let c_root = splice(&mut builder, &mut env, c.root(), &c.inner.env);
        let root = builder.push_ternary(op, a_root, b_root, c_root);
        let mut buffers = self.inner.buffers.clone();
        merge_buffer_data(&mut buffers, &b.inner.buffers);
        merge_buffer_data(&mut buffers, &c.inner.buffers);
        Self::wrap(builder.finish(&[root]), env, buffers)
    }

    // ───────────────────────── arithmetic ─────────────────────────

    /// `self + rhs`.
    #[must_use]
    pub fn add(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Add)
    }
    /// `self - rhs`.
    #[must_use]
    pub fn sub(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Sub)
    }
    /// `self * rhs`.
    #[must_use]
    pub fn mul(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Mul)
    }
    /// `self / rhs`.
    #[must_use]
    pub fn div(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Div)
    }
    /// `min(self, rhs)`.
    #[must_use]
    pub fn min(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Min)
    }
    /// `max(self, rhs)`.
    #[must_use]
    pub fn max(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Max)
    }

    /// `-self`.
    #[must_use]
    pub fn neg(&self) -> Self {
        self.map(OpKind::Neg)
    }
    /// `|self|`.
    #[must_use]
    pub fn abs(&self) -> Self {
        self.map(OpKind::Abs)
    }
    /// `√self`.
    #[must_use]
    pub fn sqrt(&self) -> Self {
        self.map(OpKind::Sqrt)
    }
    /// `1/self`.
    #[must_use]
    pub fn recip(&self) -> Self {
        self.map(OpKind::Recip)
    }
    /// `1/√self`.
    #[must_use]
    pub fn rsqrt(&self) -> Self {
        self.map(OpKind::Rsqrt)
    }

    // ────────────────────────── rounding ──────────────────────────

    /// `⌊self⌋`.
    #[must_use]
    pub fn floor(&self) -> Self {
        self.map(OpKind::Floor)
    }
    /// `⌈self⌉`.
    #[must_use]
    pub fn ceil(&self) -> Self {
        self.map(OpKind::Ceil)
    }
    /// Round to the nearest integer.
    #[must_use]
    pub fn round(&self) -> Self {
        self.map(OpKind::Round)
    }
    /// The fractional part, `self - ⌊self⌋`. Library, not a primitive.
    #[must_use]
    pub fn fract(&self) -> Self {
        self.sub(&self.floor())
    }

    // ───────────────────── transcendentals ────────────────────────

    /// `sin self` (radians).
    #[must_use]
    pub fn sin(&self) -> Self {
        self.map(OpKind::Sin)
    }
    /// `cos self` (radians).
    #[must_use]
    pub fn cos(&self) -> Self {
        self.map(OpKind::Cos)
    }
    /// `tan self` (radians).
    #[must_use]
    pub fn tan(&self) -> Self {
        self.map(OpKind::Tan)
    }
    /// `asin self`.
    #[must_use]
    pub fn asin(&self) -> Self {
        self.map(OpKind::Asin)
    }
    /// `acos self`.
    #[must_use]
    pub fn acos(&self) -> Self {
        self.map(OpKind::Acos)
    }
    /// `atan self`.
    #[must_use]
    pub fn atan(&self) -> Self {
        self.map(OpKind::Atan)
    }
    /// `atan2(self, x)` — the quadrant-correct angle, i.e. the polar angle of
    /// `(x, self)`.
    #[must_use]
    pub fn atan2(&self, x: &Kernel) -> Self {
        self.combine(x, OpKind::Atan2)
    }
    /// `e^self`.
    #[must_use]
    pub fn exp(&self) -> Self {
        self.map(OpKind::Exp)
    }
    /// `2^self`.
    #[must_use]
    pub fn exp2(&self) -> Self {
        self.map(OpKind::Exp2)
    }
    /// `ln self`.
    #[must_use]
    pub fn ln(&self) -> Self {
        self.map(OpKind::Ln)
    }
    /// `log₂ self`.
    #[must_use]
    pub fn log2(&self) -> Self {
        self.map(OpKind::Log2)
    }
    /// `self^exponent`.
    #[must_use]
    pub fn pow(&self, exponent: &Kernel) -> Self {
        self.combine(exponent, OpKind::Pow)
    }
    /// `√(self² + other²)` — the length of `(self, other)`. Library, not a
    /// primitive: no hardware computes it, so a `Hypot` node bought nothing
    /// but a decomposition each backend had to write for itself.
    #[must_use]
    pub fn hypot(&self, other: &Kernel) -> Self {
        self.mul(self).add(&other.mul(other)).sqrt()
    }

    // ─────────────────────── comparisons / masks ──────────────────

    /// `self < rhs` (a mask).
    #[must_use]
    pub fn lt(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Lt)
    }
    /// `self <= rhs`.
    #[must_use]
    pub fn le(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Le)
    }
    /// `self > rhs`.
    #[must_use]
    pub fn gt(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Gt)
    }
    /// `self >= rhs`.
    #[must_use]
    pub fn ge(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::Ge)
    }
    /// Mask AND (canonical masks in both tiers).
    #[must_use]
    pub fn and(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::BitAnd)
    }
    /// Mask OR.
    #[must_use]
    pub fn or(&self, rhs: &Kernel) -> Self {
        self.combine(rhs, OpKind::BitOr)
    }

    // ─────────────────────────── int domain ───────────────────────

    /// Truncate toward zero to a lane-wide `i32`, entering the BIT domain
    /// (`cvttps2dq` / `fcvtzs`).
    ///
    /// Returns [`Bits`], not `Kernel`, because the result is a bit pattern
    /// rather than a number: multiplying it by `2.0` is meaningless, and with
    /// one shared type that mistake compiles and yields plausible pixels. The
    /// type is the enforcement — see "Denote before you build" in CLAUDE.md.
    #[must_use]
    pub fn trunc_to_int(&self) -> Bits {
        Bits {
            inner: self.map(OpKind::TruncToInt),
        }
    }

    // ─────────────────────────── control ──────────────────────────

    /// `self ? if_true : if_false` — `self` is the mask.
    #[must_use]
    pub fn select(&self, if_true: &Kernel, if_false: &Kernel) -> Self {
        self.combine3(if_true, if_false, OpKind::Select)
    }
    /// `clamp(self, lo, hi)` = `min(max(self, lo), hi)`.
    ///
    /// Library, not a primitive: this builds the composition it denotes, so
    /// there is exactly one definition of clamping and every tier evaluates
    /// the same nodes. (It used to be an IR node that three backends and the
    /// e-graph's derivative rule each re-decomposed by hand, and they
    /// disagreed on degenerate `lo > hi` bounds.) Passing `lo > hi` yields
    /// `hi`, as the composition says.
    #[must_use]
    pub fn clamp(&self, lo: &Kernel, hi: &Kernel) -> Self {
        self.max(lo).min(hi)
    }

    // ───────────────────────── composition ────────────────────────

    /// `Σ kernels`, empty summing to `0` — the variadic monoid fold the
    /// fixed-arity operators cannot express (glyph outlines, text runs).
    ///
    /// Builds into ONE arena in a single pass: clone the first term's arena
    /// once, then splice each remaining term once and chain an `Add`. A naive
    /// `fold(acc.add(k))` would re-clone the *growing* accumulator arena every
    /// step — O(n²) for a glyph's thousands of leaves — so the fold is written
    /// out explicitly to stay O(total nodes).
    ///
    // DEFERRED (shared-store direction): the deeper fix is one hash-consed arena
    // that all `Kernel`s index by `ExprId`, so composition interns instead of
    // splicing (copies vanish, structural sharing is automatic). Not taken yet:
    // it changes the `Kernel` representation and wants the same store P7–P9's
    // discrete domains/typed fields will live in — land it there, deliberately,
    // rather than as a silent representation swap. The compile cache already
    // dedups at the compile boundary, so only construction-time copies remain.
    #[must_use]
    pub fn sum(kernels: &[Kernel]) -> Self {
        Self::fold(Monoid::SUM, kernels)
    }

    /// `⊕ kernels` under any [`Monoid`], empty folding to its identity.
    ///
    /// The general form of [`Kernel::sum`], which is this at [`Monoid::SUM`].
    /// What `sum` and a variadic `min` have in common is not "sum and min" —
    /// it is *monoid*, and `Monoid` already names that, so there is one
    /// definition rather than one per operator. (A glyph run wanted the `MIN`
    /// instance: distances combine under `min` exactly as windings combine
    /// under `+`.)
    ///
    /// Note this is the **fixed-arity** fold over a slice of distinct terms,
    /// not [`Kernel::over`], which folds one body over a bounded index.
    #[must_use]
    pub fn fold(monoid: Monoid, kernels: &[Kernel]) -> Self {
        let op = monoid.op();
        let Some((head, tail)) = kernels.split_first() else {
            return Self::constant(monoid.identity());
        };
        let mut b = Builder::new();
        let mut env = head.inner.env.clone();
        let mut root = copy_subgraph(&mut b, head.root());
        let mut buffers = head.inner.buffers.clone();
        for k in tail {
            let rhs = splice(&mut b, &mut env, k.root(), &k.inner.env);
            root = b.push_binary(op, root, rhs);
            merge_buffer_data(&mut buffers, &k.inner.buffers);
        }
        Self::wrap(b.finish(&[root]), env, buffers)
    }

    /// `⊕_{i ∈ 0..extent} body(i)` — **the** reduction binder: fold `body` over
    /// a bounded discrete domain under `monoid`, eliminating that dimension.
    ///
    /// This is the primitive; [`Kernel::sum_over`] and friends are one-line
    /// helpers over it, and a new [`Monoid`] extends the language without
    /// touching this method.
    ///
    /// The closure receives the bound index as a `Kernel` of its own, so Rust's
    /// scoping *is* the binder's scoping — an index cannot escape the fold that
    /// binds it, and a repeated index in nested folds is a genuine contraction
    /// rather than an accident. `extent` is a static count, which is what keeps
    /// the language total and its cost closed-form (`|D| × cost(body)`); the
    /// backend unrolls, so the domain is bounded in practice as well as in
    /// principle.
    ///
    /// Nesting is supported (up to 4 live binders — the reserved index space):
    /// each fold takes the lowest index slot its body does not already bind.
    ///
    /// ```ignore
    /// // Σ_d q(d)·k(d) — a contraction over the shared index.
    /// Kernel::over(Monoid::SUM, 64, |d| q.at_index(d).mul(&k.at_index(d)))
    /// ```
    #[must_use]
    pub fn over(monoid: Monoid, extent: u32, body: impl FnOnce(&Kernel) -> Kernel) -> Self {
        // Build the body against a placeholder index unique to this binder,
        // then rename it to a real slot once we can see which slots the body
        // already binds. Choosing the slot up-front is impossible: the body
        // (and therefore its inner binders) does not exist yet.
        let scope = BinderScope::enter();
        let index = {
            let mut b = Builder::new();
            let r = b.push_var(scope.placeholder());
            Self::wrap(b.finish(&[r]), Environment::new(), BTreeMap::new())
        };
        let body = body(&index);

        let mut b = Builder::new();
        let env = body.inner.env.clone();
        let binder = lowest_free_binder(body.dag());
        let renamed = b.push_var(binder.var());
        let body_root = substitute_vars(&mut b, body.root(), &[(scope.placeholder(), renamed)]);
        // One typed node. The encoding this replaced pushed three `Const`
        // children — combiner index, binder slot, extent — and left every
        // reader to recover them by position and by asking a float whether
        // it was really a small integer.
        let root = b.push_reduce(Fold::new(monoid, binder, 0..extent), body_root);
        // Only `body`'s own graph is used above — no other kernel is spliced
        // in — so its buffer table carries forward unchanged.
        Self::wrap(b.finish(&[root]), env, body.inner.buffers.clone())
    }

    /// `Σ_{i ∈ 0..extent} body(i)` — contraction, projection, and every other
    /// sum over a bounded index.
    #[must_use]
    pub fn sum_over(extent: u32, body: impl FnOnce(&Kernel) -> Kernel) -> Self {
        Self::over(Monoid::SUM, extent, body)
    }

    /// `Π_{i ∈ 0..extent} body(i)`.
    #[must_use]
    pub fn product_over(extent: u32, body: impl FnOnce(&Kernel) -> Kernel) -> Self {
        Self::over(Monoid::PRODUCT, extent, body)
    }

    /// `max_{i ∈ 0..extent} body(i)` — the stabilizer half of a softmax, and
    /// the shape of any "best over a bounded set" query.
    #[must_use]
    pub fn max_over(extent: u32, body: impl FnOnce(&Kernel) -> Kernel) -> Self {
        Self::over(Monoid::MAX, extent, body)
    }

    /// `min_{i ∈ 0..extent} body(i)` — e.g. the nearest hit of a bounded set
    /// of SDFs.
    #[must_use]
    pub fn min_over(extent: u32, body: impl FnOnce(&Kernel) -> Kernel) -> Self {
        Self::over(Monoid::MIN, extent, body)
    }

    /// `∃_{i ∈ 0..extent} body(i)` — a mask that is set where *any* index
    /// satisfies `body`.
    #[must_use]
    pub fn any_over(extent: u32, body: impl FnOnce(&Kernel) -> Kernel) -> Self {
        Self::over(Monoid::ANY, extent, body)
    }

    /// `∀_{i ∈ 0..extent} body(i)` — a mask that is set where *every* index
    /// satisfies `body`.
    #[must_use]
    pub fn all_over(extent: u32, body: impl FnOnce(&Kernel) -> Kernel) -> Self {
        Self::over(Monoid::ALL, extent, body)
    }

    /// Sample `self` at warped coordinates — contramap / `.at()`. Each of
    /// `cx`, `cy` is itself a kernel of the outer coordinates; the inner's
    /// `X`/`Y` are substituted by them.
    ///
    /// Two coordinates, because a lattice has two axes. A scalar that was a
    /// third or fourth coordinate is a [`Uniform`], and it needs no
    /// substitution: it is already the same value everywhere.
    #[must_use]
    pub fn at(&self, cx: &Kernel, cy: &Kernel) -> Self {
        // A warp substitutes the receiver's *coordinate* variables, and a
        // `Ref` is opaque to substitution — it has no `Var` to rewrite here,
        // only a name — so leaving one in place would sample the referent at
        // the outer coordinates and produce plausible, wrong pixels. Expand
        // first. Identity (a clone) when there is no reference to expand;
        // when there is, this is the static-linking direction taken, which
        // is the only direction that exists yet.
        let linked = self.linked();
        let mut b = Builder::new();
        let mut env = linked.inner.env.clone();
        let x = splice(&mut b, &mut env, cx.root(), &cx.inner.env);
        let y = splice(&mut b, &mut env, cy.root(), &cy.inner.env);
        let root = substitute_vars(&mut b, linked.root(), &[(0, x), (1, y)]);
        let mut buffers = linked.inner.buffers.clone();
        merge_buffer_data(&mut buffers, &cx.inner.buffers);
        merge_buffer_data(&mut buffers, &cy.inner.buffers);
        Self::wrap(b.finish(&[root]), env, buffers)
    }

    /// The derivative `∂self/∂var` (0=X, 1=Y), resolved symbolically at
    /// compile time. The building block of screen-space antialiasing: no jet
    /// domain, just an expression the calculus differentiates.
    ///
    /// # Panics
    ///
    /// Panics unless `var` names a coordinate axis.
    #[must_use]
    pub fn dwrt(&self, var: u8) -> Self {
        assert!(
            (var as usize) < crate::arena::COORD_AXES,
            "Kernel::dwrt: no axis {var}; a lattice has {} \
             (0 = X, 1 = Y)",
            crate::arena::COORD_AXES
        );
        let mut b = Builder::new();
        let r = copy_subgraph(&mut b, self.root());
        let v = b.push_const(f32::from(var));
        let root = b.push_binary(OpKind::Dwrt, r, v);
        Self::wrap(
            b.finish(&[root]),
            self.inner.env.clone(),
            self.inner.buffers.clone(),
        )
    }

    /// `∂self/∂X`.
    #[must_use]
    pub fn dx(&self) -> Self {
        self.dwrt(0)
    }
    /// `∂self/∂Y`.
    #[must_use]
    pub fn dy(&self) -> Self {
        self.dwrt(1)
    }

    // ───────────────────────── back end ───────────────────────────

    /// The underlying fragment — for the lattice bake and inspection. Not part
    /// of the composition surface; consumers use the methods above.
    #[must_use]
    pub fn parts(&self) -> (&ExprArena, ExprId) {
        (&self.inner.legacy.0, self.inner.legacy.1)
    }

    // ────────────────────────── linking ───────────────────────────

    /// This kernel as a one-node *reference* to itself: intern it in the
    /// [`KernelStore`](crate::store::KernelStore) and return the kernel
    /// `Ref(key)`.
    ///
    /// Composition splices — `a.add(&b)` copies `b`'s arena into the result —
    /// which is static linking with no alternative, and is why a 26-character
    /// string cost millions of nodes to *construct*. A reference is the other
    /// option: `k.by_ref()` denotes exactly what `k` denotes, at one node,
    /// and composes like any other kernel
    /// (docs/plans/2026-09-09-composition-is-linking.md §1).
    ///
    /// The tabulations `k` carries come along, so the data still travels with
    /// the value; the referent's arena, root and tables are reachable through
    /// the key. Today every reference is inlined again by
    /// [`expand_refs`](crate::passes::expand_refs) before anything else sees
    /// it — the linker only inlines — so this changes what a kernel *costs to
    /// build*, never what it means.
    ///
    /// This is the only way a `Ref` node is produced — so it is also why a
    /// `no_std` build cannot hold one: the store a name is looked up in needs
    /// a lock, `core` has none, and this method goes with the store behind
    /// the `std` feature. Composition by value is unaffected, being what
    /// every combinator did before a name existed.
    ///
    /// # Panics
    ///
    /// Panics on a kernel that is still *open* — one holding a
    /// [`BinderScope`] placeholder, i.e. the index a `Kernel::over` body is
    /// being built against. A name for an open term means nothing: the
    /// referent's value depends on a binding the store cannot carry, and the
    /// binder's rename cannot reach through a name to substitute it, so what
    /// expansion would put back is an index nothing binds.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn by_ref(&self) -> Self {
        let (arena, root) = self.parts();
        let open = arena.free_var_at_or_above(root, PLACEHOLDER_BASE as u8);
        assert!(
            open.is_none(),
            "Kernel::by_ref: this kernel holds Var({}), a reduction binder's \
             placeholder — it is the body of a `Kernel::over` still under \
             construction, and an open term has no identity to name it by",
            open.unwrap_or_default(),
        );
        let key = crate::store::KernelStore::intern(self);
        let mut b = Builder::new();
        let r = b.push_ref(key);
        Self::wrap(
            b.finish(&[r]),
            self.inner.env.clone(),
            self.inner.buffers.clone(),
        )
    }

    /// This kernel with every [`Ref`](ExprData::Ref) resolved to its
    /// referent — the static-linking direction of `Ref(k) ⟷ body(k)`, and
    /// the only direction that exists yet.
    ///
    /// A cheap clone when there is no reference to expand, which is the
    /// common case; the scan is over this kernel's own nodes, not its
    /// referents'.
    #[must_use]
    fn linked(&self) -> Self {
        if !self.dag().iter().any(|n| matches!(*n, ExprData::Ref(_))) {
            return self.clone();
        }
        let (arena, root) = self.parts();
        let (expanded, expanded_root) = crate::passes::expand_refs_owned(arena, root);
        let (rooted, env) = Rooted::unmarshal(&expanded, &[expanded_root]);
        Self::wrap(rooted, env, self.inner.buffers.clone())
    }

    // ────────────────────── bound-memory link ──────────────────────

    /// Seed this kernel's own tabulation table with `data` under `id` — the
    /// write side of "the data travels with the value"
    /// (docs/plans/2026-09-09-composition-is-linking.md §6). A kernel over
    /// bound memory (`DiscreteManifold::kernel`, `BilinearSampler::kernel`)
    /// calls this once, at the point its data exists, so every later
    /// composition — `.at`, `.add`, `Kernel::sum`, a reduction body, … —
    /// carries the tabulation forward with no caller gathering a binding by
    /// hand.
    ///
    /// `id` need not already be a buffer this kernel's own arena declares:
    /// the table is independent of arena structure — it travels with
    /// whichever fragment reads that identity once the two are composed
    /// together — which is what lets a table be seeded on a leaf kernel
    /// before it is spliced into a larger one.
    ///
    /// # Panics
    ///
    /// Panics if `id` already names different data in this kernel's table
    /// (see [`Kernel`]'s combinators, which enforce the same invariant on
    /// every merge).
    #[must_use]
    pub fn with_buffer_data(&self, id: BufferIdentity, data: Arc<[f32]>) -> Self {
        let mut buffers = self.inner.buffers.clone();
        merge_buffer_data(&mut buffers, &BTreeMap::from([(id, data)]));
        Self::wrap(self.inner.rooted.clone(), self.inner.env.clone(), buffers)
    }

    /// The tabulations this kernel carries, by the [`BufferIdentity`] each
    /// was seeded under ([`Kernel::with_buffer_data`]) — every buffer a
    /// consumer (`Manifold::compile`) can bind without gathering a binding
    /// separately from the kernel that reads it.
    pub fn buffer_data(&self) -> impl Iterator<Item = (BufferIdentity, &Arc<[f32]>)> {
        self.inner.buffers.iter().map(|(id, data)| (*id, data))
    }
}

/// A kernel whose lanes are BIT PATTERNS rather than numbers — the discrete
/// half of the language, entered by [`Kernel::trunc_to_int`].
///
/// [`Kernel`] carries continuous values; `Bits` carries the integer/bitwise
/// domain. Keeping them apart is load-bearing rather than tidy: with a single
/// type, `Kernel::x().shl(8)` shifts an IEEE-754 *representation* and
/// `trunc_to_int().mul(2.0)` does float arithmetic on an `i32` bit pattern.
/// Both compile, neither is meaningful, and both produce plausible-looking
/// output — the worst failure mode there is. Only the operations that are
/// meaningful on bit patterns exist here.
///
/// This types the *conversion* boundary. Comparison masks are also bit
/// patterns and still travel as `Kernel` — typing those too is the general
/// case, tracked separately.
#[derive(Clone)]
pub struct Bits {
    inner: Kernel,
}

impl Bits {
    /// `self << bits` — a logical shift of the lane's bit pattern.
    ///
    /// The count is pushed as a `Const` operand, which the emitter folds into
    /// the hardware shift immediate.
    ///
    /// # Panics
    ///
    /// Panics if `bits >= 32` — a 32-bit lane has no bits there.
    #[must_use]
    pub fn shl(&self, bits: u32) -> Self {
        assert!(bits < 32, "Bits::shl: shift of {bits} on a 32-bit lane");
        let mut b = Builder::new();
        let r = copy_subgraph(&mut b, self.inner.root());
        let count = b.push_const(bits as f32);
        let root = b.push_binary(OpKind::Shl, r, count);
        Self {
            inner: Kernel::wrap(
                b.finish(&[root]),
                self.inner.inner.env.clone(),
                self.inner.inner.buffers.clone(),
            ),
        }
    }

    /// Bitwise OR of two lane patterns.
    #[must_use]
    pub fn or(&self, rhs: &Self) -> Self {
        Self {
            inner: self.inner.or(&rhs.inner),
        }
    }

    /// Bitwise AND of two lane patterns.
    #[must_use]
    pub fn and(&self, rhs: &Self) -> Self {
        Self {
            inner: self.inner.and(&rhs.inner),
        }
    }

    /// `mask ? if_true : if_false` — a choice between two lane patterns.
    ///
    /// The same IR node as [`Kernel::select`], because there was never a
    /// second one to write: `Select` is a bitwise blend on every backend
    /// (`andps`/`andnps`/`orps`, `vpternlogd 0xCA`, `BSL`), so a choice
    /// between patterns is the instruction that already exists. What is new
    /// is the type, and it is the whole point — a colour packed into a word
    /// can now be chosen as ONE value rather than a channel at a time.
    ///
    /// An associated function, not a method, and the mask stays a [`Kernel`]:
    /// a comparison mask is a bit pattern that still travels as a `Kernel`
    /// (see this type's docs), so `mask.select(..)` is already taken and means
    /// a choice between *numbers*. Naming the domain of the arms — which is
    /// the domain of the result — is what says which of the two this is.
    #[must_use]
    pub fn select(mask: &Kernel, if_true: &Self, if_false: &Self) -> Self {
        Self {
            inner: mask.select(&if_true.inner, &if_false.inner),
        }
    }

    /// Reinterpret as a [`Kernel`] for storage or as a kernel root.
    ///
    /// The lanes still hold a bit pattern; this is the deliberate, named exit
    /// from the bit domain, taken when the pattern IS the output (a packed
    /// pixel written to a frame buffer).
    #[must_use]
    pub fn into_kernel(self) -> Kernel {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The count is still checked at runtime; the OPERAND no longer needs
    /// checking, because `Kernel::x().shl(32)` does not compile at all now —
    /// `shl` exists only on [`Bits`], which only `trunc_to_int` produces.
    #[test]
    #[should_panic(expected = "32-bit lane")]
    fn shl_past_the_lane_is_refused() {
        let _refused = Kernel::x().trunc_to_int().shl(32);
    }

    /// A hand-built arena that names the retired Z axis is refused where it
    /// would become a kernel. `Var` still carries reduction indices and the
    /// rewrite tier's pattern metavariables, so the arena cannot refuse the
    /// node itself; this is the boundary where it means a coordinate.
    #[test]
    #[should_panic(expected = "which was the Z coordinate")]
    fn an_arena_naming_a_retired_axis_is_not_a_kernel() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let z = a.push_var(2);
        let root = a.push_binary(OpKind::Add, x, z);
        let _refused = Kernel::from_parts(a, root);
    }

    /// And the same for W, so neither index is quietly readmitted.
    #[test]
    #[should_panic(expected = "which was the W coordinate")]
    fn the_fourth_axis_is_refused_too() {
        let mut a = ExprArena::new();
        let w = a.push_var(3);
        let _refused = Kernel::from_parts(a, w);
    }

    #[test]
    fn scalar_is_chosen_by_type() {
        let u = Uniform::new(0.0);
        assert!(matches!(Scalar::from(1.5), Scalar::Const(v) if v == 1.5));
        assert!(matches!(Scalar::from(u), Scalar::Uniform(h) if h == u));
        assert_ne!(
            Uniform::new(0.0),
            Uniform::new(0.0),
            "two instances are two arguments"
        );
    }

    // ───────────── the data travels with the value ─────────────

    /// [`Kernel::with_buffer_data`] seeds an entry [`Kernel::buffer_data`]
    /// reads straight back — the write and read sides of the carried table
    /// agree, a leaf kernel needs no arena declaration to carry one, and a
    /// fresh kernel carries none.
    #[test]
    fn with_buffer_data_round_trips_through_buffer_data() {
        assert_eq!(Kernel::constant(0.0).buffer_data().count(), 0);

        let id = BufferIdentity::mint();
        let data: Arc<[f32]> = Arc::from([1.0f32, 2.0, 3.0].as_slice());
        let seeded = Kernel::constant(0.0).with_buffer_data(id, Arc::clone(&data));

        let found: Vec<_> = seeded.buffer_data().collect();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, id);
        assert!(Arc::ptr_eq(found[0].1, &data));
    }

    /// Every combinator that splices a second kernel in — `combine`
    /// (binary arithmetic/comparison), `combine3` (`select`), `sum`, and
    /// `at` — carries a seeded tabulation through to the result, and `map`/
    /// `dwrt`/`over`, which touch only one kernel's own arena, carry it
    /// forward unchanged. One test per chokepoint, all against the same
    /// seeded leaf, so a combinator that forgot to merge shows up as an
    /// empty `buffer_data()` rather than a wrong pixel three call sites away.
    #[test]
    fn every_combinator_carries_a_seeded_tabulation_through() {
        let id = BufferIdentity::mint();
        let data: Arc<[f32]> = Arc::from([7.0f32].as_slice());
        let seeded = Kernel::constant(1.0).with_buffer_data(id, data);
        let plain = Kernel::constant(2.0);

        let carries = |k: &Kernel| k.buffer_data().any(|(found, _)| found == id);

        assert!(carries(&seeded.add(&plain)), "combine (binary) via add");
        assert!(carries(&plain.add(&seeded)), "combine (binary), rhs seeded");
        assert!(
            carries(&Kernel::x().select(&seeded, &plain)),
            "combine3 via select's if_true arm"
        );
        assert!(
            carries(&Kernel::x().select(&plain, &seeded)),
            "combine3 via select's if_false arm"
        );
        assert!(
            carries(&Kernel::sum(&[plain.clone(), seeded.clone()])),
            "sum"
        );
        assert!(
            carries(&plain.at(&seeded, &Kernel::y())),
            "at, seeded in the X contramap"
        );
        assert!(carries(&seeded.map(OpKind::Neg)), "map touches only self");
        assert!(carries(&seeded.dwrt(0)), "dwrt touches only self");
        assert!(
            carries(&Kernel::sum_over(3, |i| i.add(&seeded))),
            "over carries its body's table"
        );
    }

    /// Two kernels that read the SAME buffer — the common case of a repeated
    /// glyph in a run, or a texture sampled from two places — merge into one
    /// entry rather than two, because both name the identity with the very
    /// same `Arc`.
    #[test]
    fn two_reads_of_one_identity_merge_into_one_entry() {
        let id = BufferIdentity::mint();
        let data: Arc<[f32]> = Arc::from([1.0f32].as_slice());
        let left = Kernel::x().with_buffer_data(id, Arc::clone(&data));
        let right = Kernel::y().with_buffer_data(id, Arc::clone(&data));

        let merged = left.add(&right);
        assert_eq!(merged.buffer_data().count(), 1);
    }

    /// The other side of that merge: two DIFFERENT tabulations claiming the
    /// same identity is a programming error `BufferIdentity::mint`'s own
    /// contract rules out by construction, so a combinator asserts rather
    /// than silently keeping one arm's data and discarding the other's.
    #[test]
    #[should_panic(expected = "names two different tabulations")]
    fn two_different_tabulations_under_one_identity_is_refused() {
        let id = BufferIdentity::mint();
        let left = Kernel::x().with_buffer_data(id, Arc::from([1.0f32].as_slice()));
        let right = Kernel::y().with_buffer_data(id, Arc::from([1.0f32].as_slice()));
        let _refused = left.add(&right);
    }
}
