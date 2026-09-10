//! IR-to-IR transforms: legalization.
//!
//! Five passes, each `(arena, root) -> (arena, root)`, each turning nodes no
//! backend can emit into nodes every backend can:
//!
//! | pass | consumes | produces |
//! |---|---|---|
//! | [`expand_refs`] | `Ref` | the referent, spliced in |
//! | [`lower_dwrt`] | `Dwrt` | arithmetic, and *re-introduces* transcendentals |
//! | [`expand_reduce`] | `Reduce` | the combiner applied over unrolled copies |
//! | [`expand_gather`] | `Gather` | index arithmetic + `RawGather` |
//! | [`expand_transcendentals`] | `Sin`..`Pow` | arithmetic + bit-manip atoms |
//!
//! The order in that table is the order they must run: differentiating a `sin`
//! produces a `cos`, so `lower_dwrt` has to go before the pass that expands
//! them, and you cannot differentiate a *name*, so `expand_refs` goes before
//! everything. Every pass is idempotent and has an identity fast-path, so
//! running one that has nothing to do is free.
//!
//! **Nothing here knows what it is lowering *for*.** There is no `cfg` in this
//! module beyond `#[cfg(test)]`, and no import outside `crate::{arena, kind,
//! variance}`. The legal set happens to be uniform across the backends today;
//! if it stops being uniform, that belongs in a target description these passes
//! consult, not in a `cfg` here.
//!
//! On transcendentals specifically: `sin`, `cos`, `atan` have no single
//! instruction on any target — they are *always* a polynomial — so they are not
//! a backend's business. Expanding them here means no emitter ever contains
//! transcendental assembly, the polynomial has one home, and precision is a
//! property of this code rather than of whichever backend you landed on.
//! The expansions deliberately avoid `MulAdd` and `Select`, staying inside the
//! differentiable primitive set so `lower_dwrt` can still get through them.
//!
//! They may use `Select`. [`legalize`] runs `lower_dwrt` *before*
//! `expand_transcendentals` — the chain rule manufactures `Sin`/`Cos` nodes
//! that the transcendental pass must still lower — so an expansion is only ever
//! evaluated, never differentiated, and derivatives are taken against the
//! symbolic rules in `diff_node` instead. (`lower_dwrt` carries a `Select` rule
//! regardless: it blends the branch derivatives on the primal mask.)
//!
//! Nothing re-fuses `mul`+`add` into `MulAdd` afterwards — see `horner_step`.

use crate::arena::{ExprArena, ExprId, ExprNode};
use crate::fold::Fold;
use crate::kind::OpKind;
use crate::variance::Variance;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

/// Run every legalization pass, in the one order they compose in.
///
/// This is the whole pipeline. It was previously four calls copied into each
/// compile entry, which is how two since-deleted entries came to run none of
/// them, and how the deleted
/// `CompileWorkspace` came to run none of them *and* skip the guard that
/// refuses a surviving `Dwrt`. An order that has to be retyped is an order
/// that can be forgotten.
///
/// Every pass has an identity fast-path, so calling this on an arena that
/// needs nothing lowered costs four comparisons and no allocation. There is
/// no reason for a caller to want a subset.
///
/// # Errors
///
/// Propagates [`lower_dwrt_owned`]'s error for expressions with no derivative
/// rule — bound-memory reads, integer/bit ops, reductions.
pub fn legalize(arena: &ExprArena, root: ExprId) -> Result<(ExprArena, ExprId), &'static str> {
    // `expand_refs` before anything else: every pass below reads structure,
    // and a reference has none to read — you cannot differentiate a name.
    let (arena, root) = expand_refs_owned(arena, root);
    // `lower_dwrt` next: differentiating a `sin` manufactures a `cos`, so it
    // has to precede the pass that expands them.
    let (arena, root) = lower_dwrt_owned(&arena, root)?;
    let (arena, root) = expand_reduce_owned(&arena, root);
    let (arena, root) = expand_gather_owned(&arena, root);
    Ok(expand_transcendentals_owned(&arena, root))
}

/// Whether `op` is a unary transcendental this pass expands.
fn is_transcendental_unary(op: OpKind) -> bool {
    matches!(
        op,
        OpKind::Sin
            | OpKind::Cos
            | OpKind::Tan
            | OpKind::Exp
            | OpKind::Exp2
            | OpKind::Ln
            | OpKind::Log2
            | OpKind::Log10
            | OpKind::Atan
            | OpKind::Asin
            | OpKind::Acos
    )
}

/// Whether `op` is a binary transcendental this pass expands.
fn is_transcendental_binary(op: OpKind) -> bool {
    matches!(op, OpKind::Atan2 | OpKind::Pow)
}

/// Post-order rebuild of the arena reachable from `root`, one lowering pass.
///
/// For each node (children first), `lower(arena, node, map)` may return
/// `Some(new)` to replace it — using `map(old_child)` to look up an
/// already-lowered child — or `None` to keep it as a plain structural copy.
/// Shared subexpressions are rebuilt once (`id_map` dedups), so a DAG stays a
/// DAG. This is the single skeleton behind [`expand_transcendentals`],
/// [`expand_gather`], and [`expand_reduce`]; each supplies only its `lower`
/// hook. Mirrors [`ExprArena::substitute_params`].
fn rebuild_arena<F>(arena: &mut ExprArena, root: ExprId, mut lower: F) -> ExprId
where
    F: FnMut(&mut ExprArena, &ExprNode, &dyn Fn(ExprId) -> ExprId) -> Option<ExprId>,
{
    match try_rebuild_arena::<Never, _>(arena, root, |arena, node, m| Ok(lower(arena, node, m))) {
        Ok(id) => id,
        Err(never) => match never {},
    }
}

/// Uninhabited error type for the infallible [`rebuild_arena`] wrapper.
enum Never {}

/// Fallible core of [`rebuild_arena`]: the hook may reject a node (e.g. an
/// operator [`lower_dwrt`] cannot differentiate), aborting the whole pass.
fn try_rebuild_arena<E, F>(arena: &mut ExprArena, root: ExprId, mut lower: F) -> Result<ExprId, E>
where
    F: FnMut(&mut ExprArena, &ExprNode, &dyn Fn(ExprId) -> ExprId) -> Result<Option<ExprId>, E>,
{
    let old_len = arena.len();
    let mut id_map: Vec<Option<ExprId>> = alloc::vec![None; old_len];

    enum Task {
        Descend(ExprId),
        Emit(ExprId),
    }
    let mut work: Vec<Task> = alloc::vec![Task::Descend(root)];

    while let Some(task) = work.pop() {
        match task {
            Task::Descend(id) => {
                if id_map[id.0 as usize].is_some() {
                    continue;
                }
                work.push(Task::Emit(id));
                // Descend children reversed so they emit left-to-right.
                let children: Vec<ExprId> = arena.children(id).collect();
                for child in children.into_iter().rev() {
                    work.push(Task::Descend(child));
                }
            }
            Task::Emit(id) => {
                if id_map[id.0 as usize].is_some() {
                    continue;
                }
                let node = arena.node(id).clone();
                let m = |old: ExprId| id_map[old.0 as usize].expect("child lowered before parent");
                let new_id = match lower(arena, &node, &m)? {
                    Some(new) => new,
                    None => copy_node(arena, id, &node, &m),
                };
                id_map[id.0 as usize] = Some(new_id);
            }
        }
    }

    Ok(id_map[root.0 as usize].expect("root lowered"))
}

/// Structural copy of `node` into `arena` with its children remapped by `m`.
/// The default action for any node a lowering hook does not replace.
fn copy_node(
    arena: &mut ExprArena,
    source: ExprId,
    node: &ExprNode,
    m: &dyn Fn(ExprId) -> ExprId,
) -> ExprId {
    match node {
        ExprNode::Var(i) => arena.push_var(*i),
        ExprNode::Const(v) => arena.push_const(*v),
        ExprNode::Param(i) => arena.push_param(*i),
        // Same arena, so the buffer and uniform tables (and ids) stay valid.
        ExprNode::Buffer(b) => arena.push_buffer(*b),
        ExprNode::Uniform(u) => arena.push_uniform(*u),
        ExprNode::Ref(k) => arena.push_ref(*k),
        ExprNode::Unary(op, a) => arena.push_unary(*op, m(*a)),
        ExprNode::Binary(op, a, b) => arena.push_binary(*op, m(*a), m(*b)),
        ExprNode::Ternary(op, a, b, c) => arena.push_ternary(*op, m(*a), m(*b), m(*c)),
        ExprNode::Nary(op, ..) => {
            let mapped: Vec<ExprId> = arena.children(source).map(m).collect();
            arena.push_nary(*op, &mapped)
        }
        ExprNode::Reduce { fold, body } => arena.push_reduce(*fold, m(*body)),
    }
}

// ──────────────────────────────── Ref expansion ──────────────────────────────

/// Replace every [`ExprNode::Ref`] reachable from `root` with its referent,
/// spliced in, returning the (possibly new) root in the same arena.
///
/// This is the linker, and in this stage it only inlines
/// (docs/plans/2026-09-09-composition-is-linking.md §3): a reference is
/// resolved through the [`KernelStore`](crate::store::KernelStore) and its
/// body copied in at the reference's position, reading the same coordinates
/// the reference did. The splice merges the referent's buffer and uniform
/// declarations into this arena by identity, exactly as composition does, so
/// a referent over bound memory keeps naming the same memory.
///
/// Recursive: a referent may itself hold references, and each is expanded
/// before its body is spliced. That terminates because references form a DAG
/// by construction — a key names content that already existed when the key
/// was minted, so nothing can reference itself.
///
/// **Two uses of one name are one node.** That is what a name is for: the
/// referent is spliced once per key, and every later `Ref` to that key
/// points at the same subgraph. A kernel referenced `m` times therefore
/// costs one copy of its body rather than `m` — and a reduction inside it
/// is unrolled once by [`expand_reduce`], not `m` times. Splicing per use
/// would give back exactly what composition by value costs (measured on a
/// glyph whose winding sum is read once per boundary piece: 16k, 37k, 58k
/// legalized nodes *per piece* at 40, 73, 132 pieces — quadratic).
///
/// # Panics
///
/// Panics if a key names no interned kernel. The only producer of a `Ref` is
/// `Kernel::by_ref`, which interns before it names, so an unresolvable key is
/// a corrupt graph rather than a condition to recover from.
pub fn expand_refs(arena: &mut ExprArena, root: ExprId) -> ExprId {
    let mut spliced: BTreeMap<crate::key::KernelKey, ExprId> = BTreeMap::new();
    rebuild_arena(arena, root, |arena, node, _m| match node {
        ExprNode::Ref(key) => Some(
            *spliced
                .entry(*key)
                .or_insert_with(|| splice_referent(arena, *key)),
        ),
        _ => None,
    })
}

/// Resolve one reference and splice its (itself ref-free) body into `arena`.
#[cfg(feature = "std")]
fn splice_referent(arena: &mut ExprArena, key: crate::key::KernelKey) -> ExprId {
    let referent = crate::store::KernelStore::resolve(key).unwrap_or_else(|| {
        panic!(
            "expand_refs: {key:?} names no interned kernel — every Ref is \
             minted by Kernel::by_ref, which interns first"
        )
    });
    let (ref_arena, ref_root) = referent.parts();
    let (expanded, expanded_root) = expand_refs_owned(ref_arena, ref_root);
    arena.splice(&expanded, expanded_root)
}

/// The same, where there is no store to resolve against.
///
/// Unreachable rather than unimplemented: the store *is* the `std` feature,
/// and `Kernel::by_ref` — the only producer of a `Ref` — goes with it, so a
/// `no_std` build has no way to mint the key this would look up. Reaching
/// here means one was minted by hand through `ExprArena::push_ref`, which
/// names nothing.
#[cfg(not(feature = "std"))]
fn splice_referent(_arena: &mut ExprArena, key: crate::key::KernelKey) -> ExprId {
    panic!(
        "expand_refs: {key:?} cannot be resolved — the KernelStore is the \
         `std` feature, and so is Kernel::by_ref, so nothing here can have \
         named a kernel"
    )
}

/// Owned wrapper mirroring [`expand_transcendentals_owned`]: identity
/// fast-path when the arena holds no `Ref`, otherwise clone-and-expand.
#[must_use]
pub fn expand_refs_owned(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    if !arena.nodes().any(|n| matches!(n, ExprNode::Ref(_))) {
        return (arena.clone(), root);
    }
    let mut owned = arena.clone();
    let new_root = expand_refs(&mut owned, root);
    (owned, new_root)
}

// ───────────────────────── Transcendental expansion ──────────────────────────

/// Expand every transcendental node reachable from `root` into a primitive
/// arithmetic subgraph, returning the (possibly new) root in the same arena.
/// Non-transcendental nodes are copied unchanged (see [`rebuild_arena`]).
pub fn expand_transcendentals(arena: &mut ExprArena, root: ExprId) -> ExprId {
    rebuild_arena(arena, root, |arena, node, m| match node {
        ExprNode::Unary(op, a) if is_transcendental_unary(*op) => {
            Some(expand_unary(arena, *op, m(*a)))
        }
        ExprNode::Binary(op, a, b) if is_transcendental_binary(*op) => {
            Some(expand_binary(arena, *op, m(*a), m(*b)))
        }
        _ => None,
    })
}

/// Convenience wrapper for the public compile entry, which holds a
/// shared `&ExprArena`: clone it, expand transcendentals in the clone, and
/// return the owned arena + new root. Cheap when there are no transcendentals
/// (the clone is two `Vec`s and the walk just copies), so every entry can call
/// it unconditionally and be sure no backend — per-batch or collapse —
/// ever sees a transcendental node.
#[must_use]
pub(crate) fn expand_transcendentals_owned(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    // Identity fast-path: if there is nothing to lower, return the arena
    // unchanged. The rebuild below is not bit-identical to the input (it can
    // re-order / re-dedup nodes), which would perturb register allocation for
    // transcendental-free kernels; skipping it keeps lowering a true no-op for
    // them.
    if !arena.nodes().any(|n| match n {
        ExprNode::Unary(op, _) => is_transcendental_unary(*op),
        ExprNode::Binary(op, _, _) => is_transcendental_binary(*op),
        _ => false,
    }) {
        return (arena.clone(), root);
    }
    let mut owned = arena.clone();
    let new_root = expand_transcendentals(&mut owned, root);
    (owned, new_root)
}

// ─────────────────────────────── Gather lowering ──────────────────────────────

/// Lower every high-level `Gather(buffer, x, y)` reachable from `root` into
/// index arithmetic plus a primitive [`OpKind::RawGather`], returning the
/// (possibly new) root in the same arena.
///
/// The index expression is byte-for-byte the one `DiscreteManifold::eval`
/// computes — `clamp(floor(idx), 0, extent-1)` per axis, then
/// `yi * width + xi` — so the emitter only ever sees ops it already supports
/// (`Floor`, `Clamp`, `Mul`, `Add`) plus the single `RawGather` primitive.
/// This is the analogue of [`expand_transcendentals`] for memory reads.
pub fn expand_gather(arena: &mut ExprArena, root: ExprId) -> ExprId {
    rebuild_arena(arena, root, |arena, node, m| match node {
        ExprNode::Ternary(OpKind::Gather, buf, x, y) => {
            Some(lower_gather(arena, m(*buf), m(*x), m(*y)))
        }
        _ => None,
    })
}

/// Owned wrapper mirroring [`expand_transcendentals_owned`]: identity fast-path
/// when the arena has no `Gather`, otherwise clone-and-lower.
#[must_use]
pub(crate) fn expand_gather_owned(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    if !arena
        .nodes()
        .any(|n| matches!(n, ExprNode::Ternary(OpKind::Gather, _, _, _)))
    {
        return (arena.clone(), root);
    }
    let mut owned = arena.clone();
    let new_root = expand_gather(&mut owned, root);
    (owned, new_root)
}

/// Build the index arithmetic for one gather and wrap it in a `RawGather`.
///
/// `buf`/`x`/`y` are already lowered nodes in `arena`; `buf` is a `Buffer` leaf.
/// Produces `RawGather(buf, clamp(floor(y),0,h-1) * width + clamp(floor(x),0,w-1))`,
/// matching `DiscreteManifold::eval`.
fn lower_gather(arena: &mut ExprArena, buf: ExprId, x: ExprId, y: ExprId) -> ExprId {
    let decl = match arena.node(buf) {
        ExprNode::Buffer(id) => *arena.buffer_decl(*id),
        other => panic!("lower_gather: first child must be a Buffer leaf, got {other:?}"),
    };

    let zero = arena.push_const(0.0);
    let max_x = arena.push_const(decl.width.saturating_sub(1) as f32);
    let max_y = arena.push_const(decl.height.saturating_sub(1) as f32);
    let width = arena.push_const(decl.width as f32);

    // xi = clamp(floor(x), 0, width-1); yi = clamp(floor(y), 0, height-1),
    // written as the min/max composition clamp denotes — there is no `Clamp`
    // primitive to lower to.
    let fx = arena.push_unary(OpKind::Floor, x);
    let xi_lo = arena.push_binary(OpKind::Max, fx, zero);
    let xi = arena.push_binary(OpKind::Min, xi_lo, max_x);
    let fy = arena.push_unary(OpKind::Floor, y);
    let yi_lo = arena.push_binary(OpKind::Max, fy, zero);
    let yi = arena.push_binary(OpKind::Min, yi_lo, max_y);

    // idx = yi * width + xi  (float; exact for indices < 2^24, as in DiscreteManifold)
    let row = arena.push_binary(OpKind::Mul, yi, width);
    let idx = arena.push_binary(OpKind::Add, row, xi);

    arena.push_binary(OpKind::RawGather, buf, idx)
}

// ─────────────────────────────── Reduce lowering ──────────────────────────────

/// Unroll every `Reduce` reachable from `root` into an explicit accumulation
/// tree, returning the (possibly new) root in the same arena.
///
/// `Reduce([combiner, var, extent, body])` becomes
/// `combiner(body[var:=0], combiner(body[var:=1], … body[var:=N-1]))` — N
/// inlined copies of `body` with the reduction index substituted as a `Const`.
/// Because the extent is static (bound memory), each copy's gather indices
/// become constant, so the emitter folds their addresses to immediates: the
/// fold compiles to a flat, call-free, unrolled kernel. This is the reduction
/// analogue of [`expand_gather`].
pub fn expand_reduce(arena: &mut ExprArena, root: ExprId) -> ExprId {
    rebuild_arena(arena, root, |arena, node, m| match node {
        // The body is already lowered; unroll the fold over it.
        ExprNode::Reduce { fold, body } => Some(unroll_reduce(arena, *fold, m(*body))),
        _ => None,
    })
}

/// Owned wrapper mirroring [`expand_transcendentals_owned`]: identity fast-path
/// when the arena has no `Reduce`, otherwise clone-and-lower.
#[must_use]
pub fn expand_reduce_owned(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    if !arena.nodes().any(|n| matches!(n, ExprNode::Reduce { .. })) {
        return (arena.clone(), root);
    }
    let mut owned = arena.clone();
    let new_root = expand_reduce(&mut owned, root);
    (owned, new_root)
}

/// Build the unrolled accumulation for one fold whose body is already lowered.
///
/// This is [`Fold::peel`] run to exhaustion. Peeling and unrolling are the
/// same operation at different budgets — the e-graph states the first as a
/// rewrite rule, and this is what remains for a fold that survived extraction,
/// because codegen has no iteration binder to hand it to.
fn unroll_reduce(arena: &mut ExprArena, fold: Fold, body: ExprId) -> ExprId {
    // Empty domain folds to the monoid identity.
    if fold.is_empty() {
        return arena.push_const(fold.monoid().identity());
    }
    let combiner_op = fold.monoid().op();
    let var_idx = fold.binder().var();

    // Which of the body's nodes actually vary with the index. Everything else
    // is shared across all N terms rather than copied into each of them: the
    // rewrite `⊕_i (f(i) · c) = c · ⊕_i f(i)` obtained by not duplicating `c`
    // in the first place. Computed once here, before the substitutions start
    // appending; every node reachable from `body` predates that point, so the
    // table covers each id the substitution asks about.
    let variance = crate::variance::compute_arena_variance(arena);

    let term = |arena: &mut ExprArena, k: u32| {
        Substitution::new(body, var_idx, k as f32, &variance).apply(arena, body)
    };

    // Through `Fold::peel_back`, not through `fold.range()`: this loop and
    // `egraph::fold_rules::PeelFold` are the same decomposition at different
    // budgets, and sharing the method is what keeps them the same. It is not
    // ceremony — the two produced *opposite* associations while each did its
    // own range arithmetic, and an e-graph then had to spend reassociation
    // rules reaching the shape this loop produces directly.
    //
    // `peel_back` yields the indices from the top down, so they are collected
    // and consumed in reverse: the accumulator ends up on the left and the
    // chain leans `((f(lo) ⊕ f(lo+1)) ⊕ …)`. Iterative rather than recursive
    // for the reason everything here is — the trip count is a `u16` and the
    // Rust stack is not.
    let mut indices = Vec::with_capacity(fold.len() as usize);
    let mut rest = fold;
    while let Some((shorter, k)) = rest.peel_back() {
        indices.push(k);
        rest = shorter;
    }
    let mut acc = term(arena, *indices.last().expect("a non-empty fold has terms"));
    for &k in indices.iter().rev().skip(1) {
        let next = term(arena, k);
        acc = arena.push_binary(combiner_op, acc, next);
    }
    acc
}

/// One unrolled term of a fold: the body with the bound index replaced by a
/// literal step.
///
/// The variance table is what makes this cheap. A subtree that does not depend
/// on the index would be rebuilt unchanged, so it is not rebuilt at all — the
/// original node is returned and all N terms share it. That is the rewrite
/// `⊕_i (f(i) · c) = c · ⊕_i f(i)` obtained by declining to duplicate `c`.
struct Substitution<'a> {
    /// The index being replaced, and the step to replace it with.
    var: u8,
    value: f32,
    /// Variance for every node the body can reach, indexed by `ExprId`.
    variance: &'a [Variance],
    /// Rebuilt nodes, so a shared subtree is rebuilt once and stays shared.
    ///
    /// Sized to the body, not the arena: children are pushed before their
    /// parent, so nothing the body reaches has an id above the body's own.
    /// One table per term is unavoidable (each term substitutes a different
    /// value), but an arena-sized one is written in full on allocation —
    /// `None` here is not the zero pattern — and the arena grows with every
    /// term appended, so the unroll wrote O(terms × arena) bytes to produce
    /// O(terms × body) nodes.
    memo: Vec<Option<ExprId>>,
}

impl<'a> Substitution<'a> {
    fn new(body: ExprId, var: u8, value: f32, variance: &'a [Variance]) -> Self {
        Self {
            var,
            value,
            variance,
            memo: alloc::vec![None; body.0 as usize + 1],
        }
    }

    fn apply(&mut self, arena: &mut ExprArena, id: ExprId) -> ExprId {
        let idx = id.0 as usize;
        if let Some(Some(m)) = self.memo.get(idx) {
            return *m;
        }
        if self
            .variance
            .get(idx)
            .is_some_and(|v| v.is_invariant_in(self.var))
        {
            return id;
        }
        let new = match arena.node(id).clone() {
            ExprNode::Var(i) if i == self.var => arena.push_const(self.value),
            ExprNode::Var(i) => arena.push_var(i),
            ExprNode::Const(v) => arena.push_const(v),
            ExprNode::Param(i) => arena.push_param(i),
            ExprNode::Buffer(b) => arena.push_buffer(b),
            ExprNode::Uniform(u) => arena.push_uniform(u),
            // A leaf, and a closed one: a referent binds its own reduction
            // indices, so no substitution of this fold's index can reach
            // inside it.
            ExprNode::Ref(k) => arena.push_ref(k),
            ExprNode::Unary(op, a) => {
                let a = self.apply(arena, a);
                arena.push_unary(op, a)
            }
            ExprNode::Binary(op, a, b) => {
                let a = self.apply(arena, a);
                let b = self.apply(arena, b);
                arena.push_binary(op, a, b)
            }
            ExprNode::Ternary(op, a, b, c) => {
                let a = self.apply(arena, a);
                let b = self.apply(arena, b);
                let c = self.apply(arena, c);
                arena.push_ternary(op, a, b, c)
            }
            ExprNode::Nary(op, ..) => {
                let children: Vec<ExprId> = arena.children(id).collect();
                let mapped: Vec<ExprId> = children
                    .into_iter()
                    .map(|ch| self.apply(arena, ch))
                    .collect();
                arena.push_nary(op, &mapped)
            }
            // A nested fold binds a slot of its own — `lowest_free_binder`
            // never reissues a live one — so this index cannot be captured
            // and the substitution simply passes through the body.
            ExprNode::Reduce { fold, body } => {
                let body = self.apply(arena, body);
                arena.push_reduce(fold, body)
            }
        };
        if let Some(slot) = self.memo.get_mut(idx) {
            *slot = Some(new);
        }
        new
    }
}

// ─────────────────────────────── Dwrt lowering ───────────────────────────────

/// Rewrite every `Dwrt(expr, var)` reachable from `root` into the analytic
/// derivative subgraph of `expr` with respect to coordinate `var`, returning
/// the (possibly new) root in the same arena.
///
/// This is the runtime peer of the e-graph `ChainRule` (pixelflow-search):
/// same algebra, applied directly to the arena with no e-graph dependency.
/// Derivatives of piecewise ops (`Min`/`Max`/`Select`/`Clamp`/`Abs`) mirror
/// the `Jet2` forward-mode semantics in pixelflow-core — a mask on the primal
/// values selecting between branch derivatives — so a kernel differentiated
/// here matches the combinator-over-`Jet2` path within numeric tolerance.
///
/// Runs *before* [`expand_transcendentals`] (its rules produce `Sin`/`Cos`/
/// `Exp` etc., which that pass then lowers) and processes innermost `Dwrt`
/// first, so nested derivatives (`DXX` = `Dwrt(Dwrt(e, 0), 0)`) differentiate
/// an already-`Dwrt`-free subgraph.
///
/// Errors loudly on any op with no derivative rule (integer/bit ops,
/// reductions, and a bound-memory read whose *index* moves with the variable)
/// rather than silently miscompiling.
pub fn lower_dwrt(arena: &mut ExprArena, root: ExprId) -> Result<ExprId, &'static str> {
    try_rebuild_arena(arena, root, |arena, node, m| match node {
        ExprNode::Binary(OpKind::Dwrt, expr, var) => {
            let var_idx = match arena.node(m(*var)) {
                ExprNode::Const(v) => *v as u8,
                _ => return Err("lower_dwrt: Dwrt's variable operand must be a Const"),
            };
            differentiate(arena, m(*expr), var_idx).map(Some)
        }
        ExprNode::Unary(OpKind::Dwrt, _)
        | ExprNode::Ternary(OpKind::Dwrt, _, _, _)
        | ExprNode::Nary(OpKind::Dwrt, _, _) => {
            Err("lower_dwrt: malformed Dwrt node (must be Binary(expr, var))")
        }
        _ => Ok(None),
    })
}

/// Owned wrapper mirroring [`expand_transcendentals_owned`]: identity fast-path
/// when the arena has no `Dwrt`, otherwise clone-and-lower.
pub fn lower_dwrt_owned(
    arena: &ExprArena,
    root: ExprId,
) -> Result<(ExprArena, ExprId), &'static str> {
    if !arena.nodes().any(|n| {
        matches!(
            n,
            ExprNode::Unary(OpKind::Dwrt, _)
                | ExprNode::Binary(OpKind::Dwrt, _, _)
                | ExprNode::Ternary(OpKind::Dwrt, _, _, _)
                | ExprNode::Nary(OpKind::Dwrt, _, _)
        )
    }) {
        return Ok((arena.clone(), root));
    }
    let mut owned = arena.clone();
    let new_root = lower_dwrt(&mut owned, root)?;
    Ok((owned, new_root))
}

/// Build `∂(expr)/∂(Var(var))` as new nodes in `arena`, sharing the primal
/// subgraph by id. Memoized per node, so a DAG differentiates once per shared
/// subexpression (forward-mode on the DAG, like `Jet2` carries one derivative
/// lane alongside the value).
///
/// Fully iterative — no recursion over expression depth, so arbitrarily deep
/// kernels cannot overflow the stack. Two passes: (1) mark the nodes whose
/// derivative a rule actually consumes (lazy per op: `Select` masks and
/// comparison operands are never differentiated), walking an explicit stack;
/// (2) compute marked derivatives in ascending id order — the arena is
/// append-only, so children always precede parents.
///
/// Both passes touch only what `expr` reaches, and the tables are keyed
/// rather than arena-sized: a kernel holds one `Dwrt` per antialiased edge,
/// and an arena-sized table per `Dwrt` — scanned in pass 2, and zeroed on
/// allocation — made lowering quadratic in the arena while its output stayed
/// linear (measured on a 613-piece text run: 86 s, of a 1.5 M-node arena).
fn differentiate(arena: &mut ExprArena, expr: ExprId, var: u8) -> Result<ExprId, &'static str> {
    // Pass 1: mark derivative-needed nodes.
    let mut marked: BTreeSet<ExprId> = BTreeSet::new();
    let mut stack: Vec<ExprId> = alloc::vec![expr];
    while let Some(id) = stack.pop() {
        if !marked.insert(id) {
            continue;
        }
        push_deriv_children(arena.node(id), &mut stack);
    }

    // A tabulation is the one rule that asks about *dependence* rather than
    // shape, and the variance table is the answer. Computed only where a
    // tabulation is actually reached — it is a scan of the whole arena, and a
    // kernel carries one `Dwrt` per antialiased edge, so paying for it
    // unconditionally is how this pass was quadratic before.
    //
    // Computed *before* pass 2 appends: every node `diff_node` asks about is
    // primal and so predates this point.
    let reads_memory = marked.iter().any(|id| {
        matches!(
            arena.node(*id),
            ExprNode::Ternary(OpKind::Gather, _, _, _) | ExprNode::Binary(OpKind::RawGather, _, _)
        )
    });
    let table = reads_memory.then(|| crate::variance::compute_arena_variance(arena));
    let variance = table.as_deref().unwrap_or(&[]);

    // Pass 2: bottom-up compute in topological (id) order — the set iterates
    // ascending, and the arena is append-only, so children precede parents.
    let mut memo: BTreeMap<ExprId, ExprId> = BTreeMap::new();
    for id in marked {
        // Rebuilt per node because `memo` is borrowed here and written below.
        let rules = Rules {
            var,
            memo: &memo,
            variance,
        };
        let d = diff_node(arena, id, &rules)?;
        memo.insert(id, d);
    }
    Ok(*memo
        .get(&expr)
        .expect("derivative of the root was computed"))
}

/// What a derivative rule needs besides the node itself: the variable being
/// differentiated against, the children's already-computed derivatives, and —
/// for a tabulation — whether its index moves with that variable.
struct Rules<'a> {
    var: u8,
    memo: &'a BTreeMap<ExprId, ExprId>,
    /// Variance for every primal node, or empty when no tabulation is
    /// reachable and the question is never asked.
    variance: &'a [Variance],
}

impl Rules<'_> {
    /// **A tabulation is a constant wherever its index is.** Reading memory
    /// does not vary with a coordinate — only the *address* does — so the
    /// derivative of `Gather(b, i…)` is `0` exactly when no `i` mentions the
    /// variable, which is the question [`Variance`] already answers.
    ///
    /// An index that does move is still refused: differentiating through it
    /// needs the code the tabulation replaced, and a bound buffer does not
    /// carry it (docs/plans/2026-09-09-the-graph-differentiates.md §3).
    fn tabulation(&self, arena: &mut ExprArena, index: &[ExprId]) -> Result<ExprId, &'static str> {
        let moves = index.iter().any(|id| {
            !self
                .variance
                .get(id.0 as usize)
                .is_some_and(|v| v.is_invariant_in(self.var))
        });
        match moves {
            true => Err("lower_dwrt: cannot differentiate a bound-memory read"),
            false => Ok(arena.push_const(0.0)),
        }
    }
}

/// Which children's derivatives the rule for `node` consumes. Must stay in
/// lockstep with [`diff_node`]: a child pushed here is differentiated eagerly;
/// a child omitted here must not be read from the memo there. Ops with no
/// rule push nothing — [`diff_node`] raises the error for the node itself.
fn push_deriv_children(node: &ExprNode, stack: &mut Vec<ExprId>) {
    match *node {
        ExprNode::Var(_)
        | ExprNode::Const(_)
        | ExprNode::Param(_)
        | ExprNode::Buffer(_)
        | ExprNode::Uniform(_)
        | ExprNode::Ref(_) => {}
        ExprNode::Unary(op, a) => match op {
            // d = 0 without touching the operand.
            OpKind::Floor | OpKind::Ceil | OpKind::Round => {}
            // No rule: the error surfaces at the node, not its children.
            OpKind::TruncToInt | OpKind::IntToFloat => {}
            _ => stack.push(a),
        },
        ExprNode::Binary(op, a, b) => match op {
            OpKind::Add
            | OpKind::Sub
            | OpKind::Mul
            | OpKind::Div
            | OpKind::Min
            | OpKind::Max
            | OpKind::Atan2
            | OpKind::Pow => {
                stack.push(a);
                stack.push(b);
            }
            // Masks: d = 0 without touching the operands.
            OpKind::Lt | OpKind::Le | OpKind::Gt | OpKind::Ge | OpKind::Eq | OpKind::Ne => {}
            _ => {}
        },
        ExprNode::Ternary(op, a, b, c) => match op {
            OpKind::MulAdd => {
                stack.push(a);
                stack.push(b);
                stack.push(c);
            }
            // The mask is never differentiated.
            OpKind::Select => {
                stack.push(b);
                stack.push(c);
            }
            _ => {}
        },
        ExprNode::Nary(_, _, _) => {}
        // No rule: `diff_node` raises the error for the fold itself.
        ExprNode::Reduce { .. } => {}
    }
}

fn diff_node(arena: &mut ExprArena, id: ExprId, rules: &Rules) -> Result<ExprId, &'static str> {
    let Rules { var, memo, .. } = *rules;
    match arena.node(id).clone() {
        ExprNode::Var(i) => Ok(arena.push_const(if i == var { 1.0 } else { 0.0 })),
        // Constants, scalar params (baked before evaluation) and uniforms
        // (invariant across the lattice) are coordinate-independent.
        ExprNode::Const(_) | ExprNode::Param(_) | ExprNode::Uniform(_) => Ok(arena.push_const(0.0)),
        ExprNode::Buffer(_) => Err("lower_dwrt: cannot differentiate a bound-memory read"),
        // You cannot differentiate a name. Give the reference a resolvable
        // referent — `expand_refs`, which every pipeline runs before this —
        // and you differentiate the referent instead.
        ExprNode::Ref(_) => Err("lower_dwrt: cannot differentiate a Ref; run expand_refs first"),

        ExprNode::Unary(op, a) => {
            // Step functions and int-domain ops never mark their operand in
            // pass 1, so the memo read must stay behind the match.
            match op {
                // Step functions: zero derivative almost everywhere.
                OpKind::Floor | OpKind::Ceil | OpKind::Round => {
                    return Ok(arena.push_const(0.0));
                }
                OpKind::TruncToInt | OpKind::IntToFloat => {
                    return Err("lower_dwrt: cannot differentiate integer/bit-manipulation ops");
                }
                _ => {}
            }
            let du = dchild(memo, a);
            match op {
                OpKind::Neg => Ok(d_neg(arena, du)),
                // d(√u) = 0.5·rsqrt(u)·u'  (Jet2 computes the same rsqrt form).
                OpKind::Sqrt => {
                    let half = arena.push_const(0.5);
                    let rs = arena.push_unary(OpKind::Rsqrt, a);
                    let factor = arena.push_binary(OpKind::Mul, half, rs);
                    Ok(d_mul(arena, factor, du))
                }
                // d(u^-1/2) = -0.5·u^-3/2·u' = -0.5·rsqrt(u)·recip(u)·u'.
                OpKind::Rsqrt => {
                    let neg_half = arena.push_const(-0.5);
                    let rs = arena.push_unary(OpKind::Rsqrt, a);
                    let rc = arena.push_unary(OpKind::Recip, a);
                    let t = arena.push_binary(OpKind::Mul, rs, rc);
                    let factor = arena.push_binary(OpKind::Mul, neg_half, t);
                    Ok(d_mul(arena, factor, du))
                }
                // d(1/u) = -u' / u².
                OpKind::Recip => {
                    let ndu = d_neg(arena, du);
                    let u2 = arena.push_binary(OpKind::Mul, a, a);
                    Ok(arena.push_binary(OpKind::Div, ndu, u2))
                }
                // d(|u|) = (u/|u|)·u'  (Jet2's sign form; NaN at 0, as there).
                OpKind::Abs => {
                    let au = arena.push_unary(OpKind::Abs, a);
                    let sign = arena.push_binary(OpKind::Div, a, au);
                    Ok(d_mul(arena, sign, du))
                }
                OpKind::Sin => {
                    let c = arena.push_unary(OpKind::Cos, a);
                    Ok(d_mul(arena, c, du))
                }
                OpKind::Cos => {
                    let s = arena.push_unary(OpKind::Sin, a);
                    let ns = arena.push_unary(OpKind::Neg, s);
                    Ok(d_mul(arena, ns, du))
                }
                // d(tan u) = u' / cos²(u).
                OpKind::Tan => {
                    let c = arena.push_unary(OpKind::Cos, a);
                    let c2 = arena.push_binary(OpKind::Mul, c, c);
                    Ok(arena.push_binary(OpKind::Div, du, c2))
                }
                // d(asin u) = u' / √(1 − u²).
                OpKind::Asin => {
                    let s = sqrt_one_minus_sq(arena, a);
                    Ok(arena.push_binary(OpKind::Div, du, s))
                }
                // d(acos u) = −u' / √(1 − u²).
                OpKind::Acos => {
                    let s = sqrt_one_minus_sq(arena, a);
                    let q = arena.push_binary(OpKind::Div, du, s);
                    Ok(arena.push_unary(OpKind::Neg, q))
                }
                // d(atan u) = u' / (1 + u²).
                OpKind::Atan => {
                    let one = arena.push_const(1.0);
                    let u2 = arena.push_binary(OpKind::Mul, a, a);
                    let den = arena.push_binary(OpKind::Add, one, u2);
                    Ok(arena.push_binary(OpKind::Div, du, den))
                }
                OpKind::Exp => {
                    let e = arena.push_unary(OpKind::Exp, a);
                    Ok(d_mul(arena, e, du))
                }
                // d(2^u) = 2^u·ln2·u'.
                OpKind::Exp2 => {
                    let e = arena.push_unary(OpKind::Exp2, a);
                    let ln2 = arena.push_const(core::f32::consts::LN_2);
                    let factor = arena.push_binary(OpKind::Mul, e, ln2);
                    Ok(d_mul(arena, factor, du))
                }
                // d(ln u) = u' / u.
                OpKind::Ln => Ok(arena.push_binary(OpKind::Div, du, a)),
                // d(log2 u) = u' / (u·ln2).
                OpKind::Log2 => {
                    let ln2 = arena.push_const(core::f32::consts::LN_2);
                    let den = arena.push_binary(OpKind::Mul, a, ln2);
                    Ok(arena.push_binary(OpKind::Div, du, den))
                }
                // d(log10 u) = u' / (u·ln10).
                OpKind::Log10 => {
                    let ln10 = arena.push_const(core::f32::consts::LN_10);
                    let den = arena.push_binary(OpKind::Mul, a, ln10);
                    Ok(arena.push_binary(OpKind::Div, du, den))
                }
                _ => Err("lower_dwrt: no derivative rule for this unary op"),
            }
        }

        ExprNode::Binary(op, a, b) => match op {
            OpKind::Add => {
                let da = dchild(memo, a);
                let db = dchild(memo, b);
                Ok(d_add(arena, da, db))
            }
            OpKind::Sub => {
                let da = dchild(memo, a);
                let db = dchild(memo, b);
                Ok(d_sub(arena, da, db))
            }
            // Product rule.
            OpKind::Mul => {
                let da = dchild(memo, a);
                let db = dchild(memo, b);
                let t1 = d_mul(arena, da, b);
                let t2 = d_mul(arena, a, db);
                Ok(d_add(arena, t1, t2))
            }
            // Quotient rule: (a'b − ab') / b².
            OpKind::Div => {
                let da = dchild(memo, a);
                let db = dchild(memo, b);
                let t1 = d_mul(arena, da, b);
                let t2 = d_mul(arena, a, db);
                let num = d_sub(arena, t1, t2);
                if is_const_zero(arena, num) {
                    return Ok(num);
                }
                let den = arena.push_binary(OpKind::Mul, b, b);
                Ok(arena.push_binary(OpKind::Div, num, den))
            }
            // Piecewise: derivative of the branch the primal takes (Jet2's
            // lt/gt masks, ties included).
            OpKind::Min => {
                let da = dchild(memo, a);
                let db = dchild(memo, b);
                let mask = arena.push_binary(OpKind::Lt, a, b);
                Ok(arena.push_ternary(OpKind::Select, mask, da, db))
            }
            OpKind::Max => {
                let da = dchild(memo, a);
                let db = dchild(memo, b);
                let mask = arena.push_binary(OpKind::Gt, a, b);
                Ok(arena.push_ternary(OpKind::Select, mask, da, db))
            }
            // Masks are step functions: zero derivative almost everywhere.
            OpKind::Lt | OpKind::Le | OpKind::Gt | OpKind::Ge | OpKind::Eq | OpKind::Ne => {
                Ok(arena.push_const(0.0))
            }
            // d(atan2(y, x)) = (x·y' − y·x') / (x² + y²).
            OpKind::Atan2 => {
                let dy = dchild(memo, a);
                let dx = dchild(memo, b);
                let t1 = d_mul(arena, b, dy);
                let t2 = d_mul(arena, a, dx);
                let num = d_sub(arena, t1, t2);
                if is_const_zero(arena, num) {
                    return Ok(num);
                }
                let y2 = arena.push_binary(OpKind::Mul, a, a);
                let x2 = arena.push_binary(OpKind::Mul, b, b);
                let den = arena.push_binary(OpKind::Add, x2, y2);
                Ok(arena.push_binary(OpKind::Div, num, den))
            }
            // d(f^g) = f^g · (g'·ln f + g·f'/f)  (Jet2's rule).
            OpKind::Pow => {
                let df = dchild(memo, a);
                let dg = dchild(memo, b);
                let lnf = arena.push_unary(OpKind::Ln, a);
                let t1 = d_mul(arena, dg, lnf);
                let g_over_f = arena.push_binary(OpKind::Div, b, a);
                let t2 = d_mul(arena, g_over_f, df);
                let inner = d_add(arena, t1, t2);
                if is_const_zero(arena, inner) {
                    return Ok(inner);
                }
                let p = arena.push_binary(OpKind::Pow, a, b);
                Ok(arena.push_binary(OpKind::Mul, p, inner))
            }
            OpKind::Dwrt => Err("lower_dwrt: nested Dwrt survived lowering (internal invariant)"),
            OpKind::RawGather => rules.tabulation(arena, &[b]),
            OpKind::IAdd | OpKind::Shl | OpKind::Shr | OpKind::BitAnd | OpKind::BitOr => {
                Err("lower_dwrt: cannot differentiate integer/bit-manipulation ops")
            }
            _ => Err("lower_dwrt: no derivative rule for this binary op"),
        },

        ExprNode::Ternary(op, a, b, c) => match op {
            // d(a·b + c) = a'·b + a·b' + c'.
            OpKind::MulAdd => {
                let da = dchild(memo, a);
                let db = dchild(memo, b);
                let dc = dchild(memo, c);
                let t1 = d_mul(arena, da, b);
                let t2 = d_mul(arena, a, db);
                let prod = d_add(arena, t1, t2);
                Ok(d_add(arena, prod, dc))
            }
            // Blend the branch derivatives on the primal mask (Jet2 select).
            OpKind::Select => {
                let db = dchild(memo, b);
                let dc = dchild(memo, c);
                Ok(arena.push_ternary(OpKind::Select, a, db, dc))
            }
            OpKind::Gather => rules.tabulation(arena, &[b, c]),
            _ => Err("lower_dwrt: no derivative rule for this ternary op"),
        },

        ExprNode::Nary(_, _, _) => Err("lower_dwrt: cannot differentiate an Nary op (Tuple)"),
        // Linearity — `d(⊕_k f) = ⊕_k d(f)` — holds for `Σ` and for nothing
        // else in the monoid set: `Π` needs the product rule, and `min`/`max`
        // are selections, not sums. The rule is not written here because
        // this lowering is a *fallback*; the place for it is the rule set,
        // where the e-graph can also decline it.
        ExprNode::Reduce { .. } => Err("lower_dwrt: no derivative rule for a bounded fold"),
    }
}

/// Read a child's already-computed derivative. Pass 1 marks exactly the
/// children each rule consumes and pass 2 runs bottom-up, so the entry is
/// always populated when the parent's rule fires.
fn dchild(memo: &BTreeMap<ExprId, ExprId>, child: ExprId) -> ExprId {
    *memo
        .get(&child)
        .expect("child derivative marked and computed before parent")
}

/// `√(1 − u²)` — shared by the asin/acos rules.
fn sqrt_one_minus_sq(arena: &mut ExprArena, u: ExprId) -> ExprId {
    let one = arena.push_const(1.0);
    let u2 = arena.push_binary(OpKind::Mul, u, u);
    let t = arena.push_binary(OpKind::Sub, one, u2);
    arena.push_unary(OpKind::Sqrt, t)
}

fn is_const_zero(arena: &ExprArena, id: ExprId) -> bool {
    matches!(arena.node(id), ExprNode::Const(v) if *v == 0.0)
}

fn is_const_one(arena: &ExprArena, id: ExprId) -> bool {
    matches!(arena.node(id), ExprNode::Const(v) if *v == 1.0)
}

// Peephole constructors for derivative arithmetic. Most leaf derivatives are
// Const(0)/Const(1); folding them here keeps the lowered graph near the size
// the e-graph `ChainRule` + algebraic cleanup would produce, without pulling
// an optimizer into pixelflow-ir.

/// `a + b`, folding the additive identity.
fn d_add(arena: &mut ExprArena, a: ExprId, b: ExprId) -> ExprId {
    if is_const_zero(arena, a) {
        return b;
    }
    if is_const_zero(arena, b) {
        return a;
    }
    arena.push_binary(OpKind::Add, a, b)
}

/// `a − b`, folding zeros.
fn d_sub(arena: &mut ExprArena, a: ExprId, b: ExprId) -> ExprId {
    if is_const_zero(arena, b) {
        return a;
    }
    if is_const_zero(arena, a) {
        return arena.push_unary(OpKind::Neg, b);
    }
    arena.push_binary(OpKind::Sub, a, b)
}

/// `a · b`, folding the annihilator and identity.
fn d_mul(arena: &mut ExprArena, a: ExprId, b: ExprId) -> ExprId {
    if is_const_zero(arena, a) || is_const_zero(arena, b) {
        return arena.push_const(0.0);
    }
    if is_const_one(arena, a) {
        return b;
    }
    if is_const_one(arena, b) {
        return a;
    }
    arena.push_binary(OpKind::Mul, a, b)
}

/// `−a`, folding zero.
fn d_neg(arena: &mut ExprArena, a: ExprId) -> ExprId {
    if is_const_zero(arena, a) {
        return a;
    }
    arena.push_unary(OpKind::Neg, a)
}

/// Expand a single transcendental unary op applied to (already-lowered) `arg`.
fn expand_unary(arena: &mut ExprArena, op: OpKind, arg: ExprId) -> ExprId {
    match op {
        OpKind::Sin => expand_sin(arena, arg),
        // cos(x) = sin(x + π/2), with the π/2 applied to the *reduced*
        // argument (see `expand_sin_phase`).
        OpKind::Cos => expand_sin_phase(arena, arg, core::f32::consts::FRAC_PI_2),
        // tan(x) = sin(x) / cos(x). Expand both so neither reaches a backend.
        OpKind::Tan => {
            let s = expand_sin(arena, arg);
            let c = expand_sin_phase(arena, arg, core::f32::consts::FRAC_PI_2);
            arena.push_binary(OpKind::Div, s, c)
        }
        OpKind::Exp2 => expand_exp2(arena, arg),
        // exp(x) = 2^(x·log2 e)
        OpKind::Exp => {
            let log2e = arena.push_const(core::f32::consts::LOG2_E);
            let scaled = arena.push_binary(OpKind::Mul, arg, log2e);
            expand_exp2(arena, scaled)
        }
        OpKind::Log2 => expand_log2(arena, arg),
        // ln(x) = log2(x)·ln 2
        OpKind::Ln => {
            let l = expand_log2(arena, arg);
            let ln2 = arena.push_const(core::f32::consts::LN_2);
            arena.push_binary(OpKind::Mul, l, ln2)
        }
        // log10(x) = log2(x)·log10 2
        OpKind::Log10 => {
            let l = expand_log2(arena, arg);
            let log10_2 = arena.push_const(core::f32::consts::LOG10_2);
            arena.push_binary(OpKind::Mul, l, log10_2)
        }
        // atan(x) = atan2(x, 1)
        OpKind::Atan => {
            let one = arena.push_const(1.0);
            expand_atan2(arena, arg, one)
        }
        // asin(x) = atan2(x, sqrt(1 - x²))
        OpKind::Asin => {
            let one = arena.push_const(1.0);
            let x2 = arena.push_binary(OpKind::Mul, arg, arg);
            let t = arena.push_binary(OpKind::Sub, one, x2);
            let s = arena.push_unary(OpKind::Sqrt, t);
            expand_atan2(arena, arg, s)
        }
        // acos(x) = atan2(sqrt(1 - x²), x)
        OpKind::Acos => {
            let one = arena.push_const(1.0);
            let x2 = arena.push_binary(OpKind::Mul, arg, arg);
            let t = arena.push_binary(OpKind::Sub, one, x2);
            let s = arena.push_unary(OpKind::Sqrt, t);
            expand_atan2(arena, s, arg)
        }
        _ => unreachable!("expand_unary called on non-transcendental {op:?}"),
    }
}

/// Expand a binary transcendental applied to (already-lowered) `a`, `b`.
fn expand_binary(arena: &mut ExprArena, op: OpKind, a: ExprId, b: ExprId) -> ExprId {
    match op {
        OpKind::Atan2 => expand_atan2(arena, a, b),
        // pow(a, b) = 2^(b·log2 a) — the same identity the backends' `pow`
        // builtins each implemented by calling their own log2/exp2 bodies.
        // Expanding here is what lets those bodies leave the assemblers.
        OpKind::Pow => {
            let l = expand_log2(arena, a);
            let scaled = arena.push_binary(OpKind::Mul, b, l);
            expand_exp2(arena, scaled)
        }
        _ => unreachable!("expand_binary called on non-transcendental {op:?}"),
    }
}

/// Degree-7 odd **minimax** coefficients for `atan(t)` on `t ∈ [-1, 1]`.
///
/// Max error 8.7e-5. The Taylor coefficients for the same degree
/// (`1, -1/3, 1/5, -1/7`) are 6.2e-2 at `|t| = 1` — 704× worse for exactly the
/// same four multiplies and three adds, because Taylor spends its accuracy
/// budget at the origin while this interval's error is dominated by the
/// endpoint. Fitting the interval instead of the point is free: `atan2` expands
/// to 27 ops either way.
///
/// The fit is over `[0, 1]`, but both `atan` and an odd polynomial are odd, so
/// the error is antisymmetric and the bound carries to `[-1, 1]` unchanged.
pub const ATAN_MINIMAX: [f32; 4] = [0.999_268_04, -0.321_431_33, 0.146_614_41, -0.039_132_48];

/// `atan2(y, x)` (four-quadrant) as a primitive subgraph.
///
/// Reduces to a ratio in [-1,1] (swapping y/x when |y|>|x|), a degree-7 odd
/// polynomial for atan on that interval, then
/// quadrant fix-ups via `Select` on comparison masks. Uses `Select`/`Lt`/`Gt`/
/// `Ge`/`Recip` — all primitives the value path emits. (Like other Select-using
/// expansions this is value-path only; the jet path has no Ternary rule.)
fn expand_atan2(arena: &mut ExprArena, y: ExprId, x: ExprId) -> ExprId {
    let pi = arena.push_const(core::f32::consts::PI);
    let half_pi = arena.push_const(core::f32::consts::FRAC_PI_2);
    let zero = arena.push_const(0.0);

    let abs_x = arena.push_unary(OpKind::Abs, x);
    let abs_y = arena.push_unary(OpKind::Abs, y);

    // swap = |y| > |x|; ratio = swap ? x/y : y/x  (keeps |ratio| <= 1).
    let swap = arena.push_binary(OpKind::Gt, abs_y, abs_x);
    let recip_y = arena.push_unary(OpKind::Recip, y);
    let recip_x = arena.push_unary(OpKind::Recip, x);
    let x_over_y = arena.push_binary(OpKind::Mul, x, recip_y);
    let y_over_x = arena.push_binary(OpKind::Mul, y, recip_x);
    let ratio = arena.push_ternary(OpKind::Select, swap, x_over_y, y_over_x);

    // atan(ratio) on [-1,1]: ratio · Horner(c7,c5,c3,c1)(ratio²).
    let r2 = arena.push_binary(OpKind::Mul, ratio, ratio);
    let mut p = arena.push_const(ATAN_MINIMAX[ATAN_MINIMAX.len() - 1]);
    for &c in ATAN_MINIMAX.iter().rev().skip(1) {
        let c = arena.push_const(c);
        p = horner_step(arena, p, r2, c);
    }
    let atan_small = arena.push_binary(OpKind::Mul, ratio, p);

    // If swapped, result is ±π/2 − atan_small (sign from ratio).
    let ratio_nonneg = arena.push_binary(OpKind::Ge, ratio, zero);
    let neg_half_pi = arena.push_unary(OpKind::Neg, half_pi);
    let signed_half = arena.push_ternary(OpKind::Select, ratio_nonneg, half_pi, neg_half_pi);
    let swapped_val = arena.push_binary(OpKind::Sub, signed_half, atan_small);
    let atan_val = arena.push_ternary(OpKind::Select, swap, swapped_val, atan_small);

    // Quadrant fix-up: if x < 0, add ±π (sign from y).
    let x_neg = arena.push_binary(OpKind::Lt, x, zero);
    let y_neg = arena.push_binary(OpKind::Lt, y, zero);
    let neg_pi = arena.push_unary(OpKind::Neg, pi);
    let adjust = arena.push_ternary(OpKind::Select, y_neg, neg_pi, pi);
    let adjusted = arena.push_binary(OpKind::Add, atan_val, adjust);
    arena.push_ternary(OpKind::Select, x_neg, adjusted, atan_val)
}

/// `2^x` as a primitive subgraph.
///
/// Split `x = xi + xf` (xi integer, xf ∈ [0,1)); approximate `2^xf` by a
/// degree-5 minimax polynomial; reconstruct `2^xi` by writing the IEEE-754
/// exponent field directly: `2^xi = bitcast((int(xi) + 127) << 23)`. Built from
/// the bit-manip primitives (`TruncToInt`/`IntToFloat`/`IAdd`/`Shl`) — these are
/// the float↔int conversions a backend cannot avoid for exp/log.
fn expand_exp2(arena: &mut ExprArena, arg_x: ExprId) -> ExprId {
    // Clamp to a safe exponent range to avoid int overflow / inf.
    let lo = arena.push_const(-EXP2_CLAMP);
    let hi = arena.push_const(EXP2_CLAMP);
    let x = arena.push_binary(OpKind::Max, arg_x, lo);
    let x = arena.push_binary(OpKind::Min, x, hi);

    // xi = floor(x), xf = x - xi
    let xi = arena.push_unary(OpKind::Floor, x);
    let xf = arena.push_binary(OpKind::Sub, x, xi);

    // 2^xf ≈ Horner([`EXP2_POLY`]) at xf, highest degree down.
    let mut p = arena.push_const(EXP2_POLY[EXP2_POLY.len() - 1]);
    for &c in EXP2_POLY.iter().rev().skip(1) {
        let c = arena.push_const(c);
        p = horner_step(arena, p, xf, c);
    }

    // 2^xi = bitcast((int(xi) + 127) << 23).
    let xi_int = arena.push_unary(OpKind::TruncToInt, xi);
    let bias = arena.push_const(f32::from_bits(127)); // integer 127 as lane bits
    let biased = arena.push_binary(OpKind::IAdd, xi_int, bias);
    // Shift amount is read by value (`v as u32 as u8`), so it is a plain 23.0.
    let shift = arena.push_const(23.0);
    let pow2i = arena.push_binary(OpKind::Shl, biased, shift); // bitcast result

    // 2^x = 2^xf · 2^xi
    let val = arena.push_binary(OpKind::Mul, p, pow2i);

    // Outside domain (NaN input), return NaN. Check original arg_x, not clamped x.
    let is_not_nan = arena.push_binary(OpKind::Eq, arg_x, arg_x);
    let nan = arena.push_const(f32::NAN);
    arena.push_ternary(OpKind::Select, is_not_nan, val, nan)
}

/// `log2(x)` as a primitive subgraph (documented domain x > 0).
///
/// Cephes `log2f` algorithm. `log2(x) = e + log2(m)` where `e` is the unbiased
/// exponent and `m ∈ [1,2)` is the mantissa. Extract `e` by shifting the
/// exponent field down; rebuild `m` by masking the mantissa bits and OR-ing in
/// exponent bias 127 (= 1.0). When `m ≥ √2`, halve `m` and bump `e` so the
/// polynomial argument `t = m − 1` stays in `[√2/2 − 1, √2 − 1]` — a degree-4
/// polynomial on the full `[1,2)` peaks at ~0.1 absolute error near `m → 2`.
/// Then `ln(1+t) = t − t²/2 + t³·P(t)` (degree-8 minimax `P`), scaled to base 2
/// via the split constant `log2 e = 1 + LOG2EA` to avoid the rounding from one
/// full-width multiply. Accurate to ~1 ulp over the reduced range.
///
/// Uses `Select` on a `Ge` mask for the range reduction, so (like the other
/// bit-manipulating expansions) this is value-path only.
fn expand_log2(arena: &mut ExprArena, x: ExprId) -> ExprId {
    // Reinterpret x's bits as int (free) and extract exponent: e = (bits >> 23) - 127.
    // Shift amount read by value -> plain 23.0.
    let shift23 = arena.push_const(23.0);
    let exp_field = arena.push_binary(OpKind::Shr, x, shift23); // int lanes
    let exp_f = arena.push_unary(OpKind::IntToFloat, exp_field);
    let bias = arena.push_const(127.0);
    let e = arena.push_binary(OpKind::Sub, exp_f, bias);

    // Mantissa m = bitcast((bits & 0x007FFFFF) | 0x3F800000) ∈ [1, 2).
    let mant_mask = arena.push_const(f32::from_bits(0x007F_FFFF));
    let one_bits = arena.push_const(f32::from_bits(0x3F80_0000));
    let mant = arena.push_binary(OpKind::BitAnd, x, mant_mask);
    let m = arena.push_binary(OpKind::BitOr, mant, one_bits);

    // Range-reduce: if m ≥ √2 { m /= 2; e += 1 } so t = m − 1 ∈ [−0.293, 0.414].
    let sqrt2 = arena.push_const(core::f32::consts::SQRT_2);
    let reduce = arena.push_binary(OpKind::Ge, m, sqrt2);
    let half = arena.push_const(0.5);
    let m_halved = arena.push_binary(OpKind::Mul, m, half);
    let m = arena.push_ternary(OpKind::Select, reduce, m_halved, m);
    let one = arena.push_const(1.0);
    let e_bumped = arena.push_binary(OpKind::Add, e, one);
    let e = arena.push_ternary(OpKind::Select, reduce, e_bumped, e);

    let t = arena.push_binary(OpKind::Sub, m, one);

    // P(t): Cephes lnf/log2f degree-8 minimax numerator for
    // (ln(1+t) − t + t²/2) / t³ on the reduced range.
    let mut p = arena.push_const(LOG2_POLY[LOG2_POLY.len() - 1]);
    for &c in LOG2_POLY.iter().rev().skip(1) {
        let c = arena.push_const(c);
        p = horner_step(arena, p, t, c);
    }

    // y = t³·P(t) − t²/2, so ln(1+t) = t + y.
    let t2 = arena.push_binary(OpKind::Mul, t, t);
    let t3 = arena.push_binary(OpKind::Mul, t2, t);
    let t3p = arena.push_binary(OpKind::Mul, t3, p);
    let half_t2 = arena.push_binary(OpKind::Mul, t2, half);
    let y = arena.push_binary(OpKind::Sub, t3p, half_t2);

    // log2(m) = (t + y)·log2(e), with log2(e) split as 1 + LOG2EA and the
    // pieces summed smallest-first (Cephes ordering) to keep full precision:
    // e + t + y + y·LOG2EA + t·LOG2EA.
    let log2ea = arena.push_const(LOG2_E_MINUS_1);
    let y_ea = arena.push_binary(OpKind::Mul, y, log2ea);
    let t_ea = arena.push_binary(OpKind::Mul, t, log2ea);
    let z = arena.push_binary(OpKind::Add, y_ea, t_ea);
    let z = arena.push_binary(OpKind::Add, z, y);
    let z = arena.push_binary(OpKind::Add, z, t);
    let val = arena.push_binary(OpKind::Add, z, e);

    // Documented domain: x > 0.0. Outside domain (x <= 0 or x is NaN), return NaN.
    // Lt(zero, x) is 0 < x (ordered comparison, false for NaN and false for x <= 0).
    let zero = arena.push_const(0.0);
    let in_domain = arena.push_binary(OpKind::Lt, zero, x);
    let nan = arena.push_const(f32::NAN);
    arena.push_ternary(OpKind::Select, in_domain, val, nan)
}

/// Largest `|x|` for which `sin`/`cos`/`tan` return a value. Beyond it they
/// return NaN — see [`expand_sin_phase`] for why the boundary is here.
///
/// `pixelflow-core`'s combinator tier re-exports this and the constants below
/// rather than restating them: it and this expansion have to be the same
/// function, and a silently drifted coefficient between them is a parity bug.
pub const TRIG_DOMAIN: f32 = 1_048_576.0; // 2^20

/// `2π` split so that `k·term` is *exact* for every `k` the domain admits.
///
/// `TAU_HI` is 25·2⁻², `TAU_MID` is 17·2⁻⁹ — 5-bit significands, so the
/// products stay exact until `|k|` reaches 2²⁴/25 ≈ 671089 and 2²⁴/17 ≈ 986895
/// respectively. [`TRIG_DOMAIN`] needs only `|k| ≤ 166887`, a 4× margin.
/// `TAU_LO` carries the remainder at full precision; the three together
/// represent `2π` to within 6.6e-13, so the reduction drifts by at most
/// `166887 · 6.6e-13 ≈ 1.1e-7` radians at the domain edge.
pub const TAU_HI: f32 = 6.25;
pub const TAU_MID: f32 = 0.033_203_125;
pub const TAU_LO: f32 = -1.781_782e-5;

/// Degree-11 odd Chebyshev coefficients for `sin(π·t)` on `t ∈ [-1, 1]`.
///
/// Degree 7 in Taylor coefficients is accurate to 7.5e-2 at the interval ends
/// — two digits — which would make the reduction work below pointless, since
/// the polynomial would then own the entire error budget.
/// Degree 9 is accurate to 6e-6 but peaks at 1.0000029: it *returns values
/// outside sin's range*, which is the defect being fixed here, so it is not an
/// option. Degree 11 is the first that both stays inside `[-1, 1]` and is
/// accurate to 6e-7, near the f32 ulp of a result near 1.
pub const SIN_CHEB: [f32; 6] = [
    3.141_591_3,
    -5.167_677_4,
    2.549_879_3,
    -0.598_278_8,
    0.080_476_06,
    -0.005_990_654,
];

/// `2^f` on `f ∈ [0, 1)`, degree-5 minimax, ascending degree.
///
/// THE definition of the exp2 polynomial. `pixelflow-core`'s backends evaluate
/// this same table so the JIT tier, the `eval_scalar` oracle and the combinator
/// tier cannot disagree about what `exp2` is — the divergence this replaced had
/// the backends on a degree-4 fit with different coefficients entirely, which
/// made `exp`, `ln`, `log10` and `pow` compute measurably different functions
/// depending on which tier ran them.
pub const EXP2_POLY: [f32; 6] = [
    1.0,
    core::f32::consts::LN_2,
    0.240_226_5,
    0.055_504_11,
    0.009_618_129,
    0.001_333_355_8,
];

/// Exponent range `exp2` saturates to, rather than overflowing to `inf`.
///
/// `2^n` is built as `bitcast((int(n) + 127) << 23)`, so an unclamped `n` past
/// this walks out of the exponent field and produces a value that is not a
/// power of two at all. CLAUDE.md lists the saturation as behavior every target
/// agrees on; clamping here is what makes that true.
pub const EXP2_CLAMP: f32 = 126.0;

/// Cephes `log2f` degree-8 minimax for `(ln(1+t) − t + t²/2) / t³` on the
/// √2-centered reduced range, ascending degree.
///
/// THE definition of the log2 polynomial, shared with `pixelflow-core`'s
/// backends for the reason [`EXP2_POLY`] gives.
pub const LOG2_POLY: [f32; 9] = [
    3.333_333e-1,
    -2.499_999_4e-1,
    2.000_071_5e-1,
    -1.666_805_8e-1,
    1.424_932_3e-1,
    -1.242_014_1e-1,
    1.167_699_9e-1,
    -1.151_461e-1,
    7.037_683_6e-2,
];

/// `log2(e) − 1`. Cephes splits `log2(e)` this way and sums the pieces
/// smallest-first to keep full precision; see [`LOG2_POLY`]'s use.
pub const LOG2_E_MINUS_1: f32 = 0.442_695_04;

/// `sin(x)` as a primitive subgraph (Chebyshev, matching the runtime path).
fn expand_sin(arena: &mut ExprArena, x: ExprId) -> ExprId {
    expand_sin_phase(arena, x, 0.0)
}

/// `sin(x + phase)` for a constant `phase`, as a primitive subgraph.
///
/// Range-reduce to `[-π, π]`, normalize to `[-1, 1]`, then [`SIN_CHEB`].
///
/// # Why the argument reduction is three terms and not one
///
/// The obvious reduction — `xx = x − k·2π` with `2π` a single f32 — is wrong
/// for large `x`, and wrong in the worst way: `k·2π` rounds to a multiple of
/// `ulp(x)`, so `xx` inherits an error that grows with `|x|` until the reduced
/// argument leaves `[-π, π]` entirely. Past that point `t` leaves `[-1, 1]`,
/// the polynomial is evaluated outside the interval it was fit on, and the
/// result diverges: `|sin|` first exceeds 1 near `x ≈ 1.4e7` and reaches `inf`
/// by `x ≈ 2.6e13`. Splitting `2π` into [`TAU_HI`]/[`TAU_MID`]/[`TAU_LO`]
/// (Cody-Waite) keeps each `k·term` exact, so the cancellation happens against
/// the true product rather than a rounded one.
///
/// # Why there is a domain limit rather than more terms
///
/// Cody-Waite buys accuracy only while `k` is exactly representable and the
/// products stay exact; extending it to the whole f32 range means Payne-Hanek
/// — a multi-word integer multiply by the bits of `1/2π` — which costs far
/// more than the polynomial it protects and is not worth it in a per-pixel
/// kernel. Past `|x| = 2²⁴` the question stops being meaningful anyway:
/// `ulp(x)` there exceeds 1 radian, so an f32 argument no longer resolves the
/// phase it is asking about.
///
/// So the reduction is honest over [`TRIG_DOMAIN`] (worst case 1.5e-6) and
/// returns NaN outside it. NaN and not a clamp into `[-1, 1]`: a clamped value
/// is a wrong answer that looks like a right one, which is exactly how this
/// defect survived — the JIT and the `eval_scalar` oracle run this same
/// expansion, so they agreed bit-for-bit on the garbage and every same-form
/// equivalence test passed.
///
/// `phase` is folded in *after* reduction rather than added to `x` up front:
/// `cos` is `sin(x + π/2)`, and at the top of the domain `ulp(x)` is 0.0625,
/// so adding π/2 to `x` there would lose most of the shift before reduction
/// ever ran.
///
/// # `sin(-0.0)` is `+0.0`, deliberately
///
/// [`TAU_LO`] is negative, so at `k = 0` the term `k·TAU_LO` is `-0.0` and the
/// last reduction step is `Sub(-0.0, -0.0)`, which is `+0.0`. Reordering the
/// three subtractions does not help — the sign dies wherever the negative
/// constant sits. Making all three positive requires `TAU_MID = 2⁻⁵`, which
/// leaves a coarser remainder and multiplies the reduction's drift by 15
/// (1.7e-6 vs 1.1e-7 at the domain edge) — that would roughly double the
/// function's total error, everywhere, to buy the sign of zero at one point.
/// Not worth it, and squarely the trade CLAUDE.md's "Floating point at the
/// edges" describes: edge-case IEEE conformance is not on offer.
fn expand_sin_phase(arena: &mut ExprArena, x: ExprId, phase: f32) -> ExprId {
    use core::f32::consts::{PI, TAU};

    let shift = |arena: &mut ExprArena, v: ExprId| {
        if phase == 0.0 {
            return v;
        }
        let p = arena.push_const(phase);
        arena.push_binary(OpKind::Add, v, p)
    };

    // k = floor((x + phase)/2π + 0.5) — the multiple of 2π nearest the
    // argument. An off-by-one in k can only happen when the argument sits on a
    // period boundary, where the two candidate reductions are ±π: the same
    // point, and sin agrees at both.
    let arg = shift(arena, x);
    let two_pi_inv = arena.push_const(1.0 / TAU);
    let half = arena.push_const(0.5);
    let u = arena.push_binary(OpKind::Mul, arg, two_pi_inv);
    let u = arena.push_binary(OpKind::Add, u, half);
    let k = arena.push_unary(OpKind::Floor, u);

    // xx = x − k·2π, in three exact pieces, then the phase back in.
    let hi = arena.push_const(TAU_HI);
    let mid = arena.push_const(TAU_MID);
    let lo = arena.push_const(TAU_LO);
    let k_hi = arena.push_binary(OpKind::Mul, k, hi);
    let k_mid = arena.push_binary(OpKind::Mul, k, mid);
    let k_lo = arena.push_binary(OpKind::Mul, k, lo);
    let xx = arena.push_binary(OpKind::Sub, x, k_hi);
    let xx = arena.push_binary(OpKind::Sub, xx, k_mid);
    let xx = arena.push_binary(OpKind::Sub, xx, k_lo);
    let xx = shift(arena, xx);

    // t = xx / π ∈ [-1, 1]. Reduction error can push |t| to ~1.03 at the
    // domain edge; SIN_CHEB still holds |p| ≤ 1 out to |t| = 1.3.
    let pi_inv = arena.push_const(1.0 / PI);
    let t = arena.push_binary(OpKind::Mul, xx, pi_inv);
    let t2 = arena.push_binary(OpKind::Mul, t, t);

    // Horner in t², expanded as mul+add.
    let mut p = arena.push_const(SIN_CHEB[SIN_CHEB.len() - 1]);
    for &c in SIN_CHEB.iter().rev().skip(1) {
        let c = arena.push_const(c);
        p = horner_step(arena, p, t2, c);
    }
    let s = arena.push_binary(OpKind::Mul, t, p);

    // Outside the domain, NaN. Guarded on the *unshifted* x so sin, cos and
    // the two halves of tan all agree about where the answer stops existing.
    // NaN itself is unguarded: |NaN| < limit is false, so it propagates.
    let limit = arena.push_const(TRIG_DOMAIN);
    let abs_x = arena.push_unary(OpKind::Abs, x);
    let in_domain = arena.push_binary(OpKind::Lt, abs_x, limit);
    let nan = arena.push_const(f32::NAN);
    arena.push_ternary(OpKind::Select, in_domain, s, nan)
}

/// `acc·x + add` as one `MulAdd` node.
///
/// Nothing downstream would fuse this for us: [`legalize`] is the last thing to
/// touch the arena, and the only thing that becomes an FMA instruction is an
/// `OpKind::MulAdd` node already in it. So a Horner chain emitted as
/// `Add(Mul(..))` reaches the emitter unfused — 5 mul + 5 add for `sin`, where
/// 5 `MulAdd`s do — and the e-graph cannot recover it either, since it runs
/// *before* this lowering and has no multiplies to fuse yet.
///
/// Fusing here rather than anywhere later is what keeps the trig identities
/// intact. Saturating after expansion would fuse these multiplies, but by then
/// there is no `Sin` node left for `AngleAddition` or `Parity` to match, and
/// the general rewriter turned loose on an expanded polynomial can reassociate
/// Horner form — which is chosen for accuracy, not just for op count.
///
/// # The cost, taken deliberately
///
/// This is the tier divergence the previous form existed to avoid, so it is
/// stated rather than discovered. Unfused mul+add rounds twice everywhere, so
/// the `eval_scalar` oracle and every backend agreed bit-for-bit. `MulAdd`
/// rounds **once** where an FMA instruction exists (x86 with `+fma`, aarch64
/// `FMLA`) and **twice** where it does not (SSE2 baseline: `mulps` + `addps`)
/// — CLAUDE.md, "Floating point at the edges". So the SSE2 tier now differs
/// from the FMA tiers by up to a rounding per Horner step.
///
/// That is a *precision* difference, which the codebase's own rule puts on the
/// table; it is not a range difference, which is not. One rounding is never
/// less accurate than two, so the FMA tiers move toward the true value, not
/// away from it, and the polynomial's range guarantees (`|sin| ≤ 1`, the
/// `TRIG_DOMAIN` NaN edge) are unaffected — they come from the reduction and
/// the `Select`, neither of which is a Horner step.
fn horner_step(arena: &mut ExprArena, acc: ExprId, x: ExprId, add: ExprId) -> ExprId {
    arena.push_ternary(OpKind::MulAdd, acc, x, add)
}

#[cfg(test)]
mod dwrt_tests {
    use super::*;
    use crate::fold::{Binder, Monoid};

    /// The first reduction binder — `Var(4)`, which these folds bind.
    fn binder() -> Binder {
        Binder::from_var(4).expect("Var(4) is the first binder")
    }

    #[test]
    fn no_dwrt_is_identity() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let e = a.push_binary(OpKind::Add, x, y);
        let (out, root) = lower_dwrt_owned(&a, e).expect("lower_dwrt");
        assert_eq!(out.len(), a.len());
        assert_eq!(root, e);
    }

    /// A pathologically deep expression must lower without stack overflow —
    /// both the rebuild and the differentiation are iterative.
    #[test]
    fn deep_chain_does_not_overflow_the_stack() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let one = a.push_const(1.0);
        let mut e = x;
        for i in 0..100_000u32 {
            e = match i % 3 {
                0 => a.push_binary(OpKind::Add, e, one),
                1 => a.push_binary(OpKind::Mul, e, x),
                _ => a.push_unary(OpKind::Sqrt, e),
            };
        }
        let v0 = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, e, v0);
        let (out, out_root) = lower_dwrt_owned(&a, root).expect("lower_dwrt");
        assert!(out.len() > a.len());
        assert!((out_root.0 as usize) < out.len());
    }

    #[test]
    fn unsupported_op_errors_loudly() {
        // Differentiating a fold has no rule here: the pass must refuse.
        let mut a = ExprArena::new();
        let body = a.push_var(4);
        let red = a.push_reduce(Fold::new(Monoid::SUM, binder(), 0..4), body);
        let v0 = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, red, v0);
        assert!(lower_dwrt_owned(&a, root).is_err());
    }

    /// A uniform is a value, never an extent — and that used to need a test,
    /// because the extent was a `Const` child and an arena could be
    /// hand-built (or *rewritten*) into holding a `Uniform` there, at which
    /// point the unroll would have read a slot index as a trip count. The
    /// extent is a field of [`Fold`] now, so there is no slot to put a
    /// uniform in and no panic left to pin. The property below — the other
    /// side of the same rule — is the half that was always about semantics.
    #[test]
    fn an_extent_is_not_an_expression() {
        let mut a = ExprArena::new();
        let body = a.push_var(4);
        let red = a.push_reduce(Fold::new(Monoid::SUM, binder(), 0..4), body);
        let ExprNode::Reduce { fold, .. } = *a.node(red) else {
            panic!("expected a fold");
        };
        assert_eq!(fold.len(), 4);
        // The trip count is not reachable from the node's children, so no
        // rewrite can substitute anything for it.
        assert_eq!(a.children(red).count(), 1);
    }

    /// `is_err()` alone can't tell a specific "no rule for this op" message
    /// apart from the generic per-arity fallback (`"no derivative rule for
    /// this {unary,binary,ternary} op"`), since both are `Err`. Assert the
    /// exact message everywhere below so a deleted specific-op arm — which
    /// falls through to the generic one — is observable.
    #[test]
    fn lower_dwrt_refuses_integer_domain_and_raw_memory_ops() {
        const BOUND_MEMORY: &str = "lower_dwrt: cannot differentiate a bound-memory read";
        const INT_BIT: &str = "lower_dwrt: cannot differentiate integer/bit-manipulation ops";

        // TruncToInt: no derivative for a discontinuous bit-reinterpret.
        // Wrapping a Gather (itself undifferentiable) distinguishes this
        // arm's own message from a leaked child error — if
        // `push_deriv_children`'s TruncToInt/IntToFloat arm wrongly marked
        // the operand as needing a derivative, the child's `BOUND_MEMORY`
        // error would surface here instead of `INT_BIT`.
        use crate::arena::{BufferDecl, BufferIdentity};
        let mut a = ExprArena::new();
        let b = a.declare_buffer(BufferDecl {
            id: BufferIdentity::mint(),
            width: 2,
            height: 1,
        });
        let gx = a.push_var(0);
        let zero = a.push_const(0.0);
        let g = a.push_gather(b, gx, zero);
        let e = a.push_unary(OpKind::TruncToInt, g);
        let v0 = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, e, v0);
        match lower_dwrt_owned(&a, root) {
            Err(msg) => assert_eq!(msg, INT_BIT),
            Ok(_) => panic!("expected {INT_BIT:?}"),
        }

        // IntToFloat, the other half of the unary integer-domain arm. Wrapped
        // around a Gather for the same reason as TruncToInt above.
        let mut a = ExprArena::new();
        let b = a.declare_buffer(BufferDecl {
            id: BufferIdentity::mint(),
            width: 2,
            height: 1,
        });
        let gx = a.push_var(0);
        let zero = a.push_const(0.0);
        let g = a.push_gather(b, gx, zero);
        let e = a.push_unary(OpKind::IntToFloat, g);
        let v0 = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, e, v0);
        match lower_dwrt_owned(&a, root) {
            Err(msg) => assert_eq!(msg, INT_BIT),
            Ok(_) => panic!("expected {INT_BIT:?}"),
        }

        // IAdd/Shl/Shr/BitAnd/BitOr: integer/bit-manipulation primitives, at
        // the binary-op level. Each is its own alternative in the grouped arm,
        // so testing only `IAdd` would let any of the other four fall through
        // to the generic per-arity fallback — a different message, or worse, a
        // derivative — while this test still passed.
        for op in [
            OpKind::IAdd,
            OpKind::Shl,
            OpKind::Shr,
            OpKind::BitAnd,
            OpKind::BitOr,
        ] {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let e = a.push_binary(op, x, y);
            let v0 = a.push_const(0.0);
            let root = a.push_binary(OpKind::Dwrt, e, v0);
            match lower_dwrt_owned(&a, root) {
                Err(msg) => assert_eq!(msg, INT_BIT, "for {op:?}"),
                Ok(_) => panic!("expected {INT_BIT:?} for {op:?}"),
            }
        }

        // A Gather whose index moves with the variable cannot be
        // differentiated, and neither can its lowered RawGather form. (An
        // index that does *not* move is a constant — see
        // `a_tabulation_is_a_constant_wherever_its_index_is`.)
        let mut a = ExprArena::new();
        let b = a.declare_buffer(BufferDecl {
            id: BufferIdentity::mint(),
            width: 2,
            height: 1,
        });
        let gx = a.push_var(0);
        let zero = a.push_const(0.0);
        let g = a.push_gather(b, gx, zero);
        let v0 = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, g, v0);
        match lower_dwrt_owned(&a, root) {
            Err(msg) => assert_eq!(msg, BOUND_MEMORY),
            Ok(_) => panic!("expected {BOUND_MEMORY:?}"),
        }

        let mut a2 = ExprArena::new();
        let b2 = a2.declare_buffer(BufferDecl {
            id: BufferIdentity::mint(),
            width: 2,
            height: 1,
        });
        let gx2 = a2.push_var(0);
        let zero2 = a2.push_const(0.0);
        let g2 = a2.push_gather(b2, gx2, zero2);
        let raw_root = expand_gather(&mut a2, g2);
        let v0b = a2.push_const(0.0);
        let root2 = a2.push_binary(OpKind::Dwrt, raw_root, v0b);
        match lower_dwrt_owned(&a2, root2) {
            Err(msg) => assert_eq!(msg, BOUND_MEMORY),
            Ok(_) => panic!("expected {BOUND_MEMORY:?}"),
        }
    }

    /// **A tabulation is a constant wherever its index is.** A piece table
    /// read at a reduce binder — the shape `fonts::loop_blinn` builds — has a
    /// coordinate-free address, so it is a number as far as X is concerned
    /// and its derivative is 0; a table read at X itself still has no
    /// derivative here, because differentiating through the address needs the
    /// code the tabulation replaced.
    ///
    /// Both halves, and both spellings (`Gather` and the `RawGather` it
    /// lowers to), because a rule that answered 0 for the second half would
    /// be a silent miscompile rather than an error.
    #[test]
    fn a_tabulation_is_a_constant_wherever_its_index_is() {
        use crate::arena::{BufferDecl, BufferIdentity, REDUCE_BINDER_BASE};

        // `index_from(&mut arena)` builds the gather's row index.
        let table = |index_from: &dyn Fn(&mut ExprArena) -> ExprId| {
            let mut a = ExprArena::new();
            let b = a.declare_buffer(BufferDecl {
                id: BufferIdentity::mint(),
                width: 4,
                height: 4,
            });
            let col = a.push_const(0.0);
            let row = index_from(&mut a);
            let g = a.push_gather(b, col, row);
            (a, b, col, row, g)
        };

        // Indexed by the reduce binder: constant in X, so `d/dX` is 0.
        let (mut a, _, _, _, g) = table(&|a| a.push_var(REDUCE_BINDER_BASE));
        let x_axis = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, g, x_axis);
        let (lowered, lroot) =
            lower_dwrt_owned(&a, root).expect("a binder-indexed read is a constant");
        assert!(
            matches!(lowered.node(lroot), ExprNode::Const(v) if *v == 0.0),
            "expected Const(0.0), got {:?}",
            lowered.node(lroot)
        );

        // The same read, one pass later: `RawGather` over the lowered address,
        // still constant in X.
        let (mut a, _, _, _, g) = table(&|a| a.push_var(REDUCE_BINDER_BASE));
        let raw = expand_gather(&mut a, g);
        let x_axis = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, raw, x_axis);
        let (lowered, lroot) =
            lower_dwrt_owned(&a, root).expect("a binder-indexed read is a constant");
        assert!(
            matches!(lowered.node(lroot), ExprNode::Const(v) if *v == 0.0),
            "expected Const(0.0), got {:?}",
            lowered.node(lroot)
        );

        // Indexed by X: the address moves, and there is no rule for that.
        let (mut a, _, _, _, g) = table(&|a| a.push_var(0));
        let x_axis = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, g, x_axis);
        match lower_dwrt_owned(&a, root) {
            Err(msg) => assert_eq!(msg, "lower_dwrt: cannot differentiate a bound-memory read"),
            Ok(_) => panic!("an X-indexed table read has no derivative here"),
        }

        // And the derivative is per-variable, not per-node: the same
        // X-indexed read is a constant in Y.
        let (mut a, _, _, _, g) = table(&|a| a.push_var(0));
        let y_axis = a.push_const(1.0);
        let root = a.push_binary(OpKind::Dwrt, g, y_axis);
        let (lowered, lroot) =
            lower_dwrt_owned(&a, root).expect("an X-indexed read is constant in Y");
        assert!(
            matches!(lowered.node(lroot), ExprNode::Const(v) if *v == 0.0),
            "expected Const(0.0), got {:?}",
            lowered.node(lroot)
        );
    }

    #[test]
    fn rebuild_copies_nary_children_slice_correctly() {
        // `copy_node`'s Nary arm reads the n-ary child slab range — a
        // second Nary node makes `start` nonzero, which is what distinguishes
        // `start+len` from `start*len` (they coincide when start is 0).
        let mut a = ExprArena::new();
        let p = a.push_var(0);
        let _throwaway = a.push_nary(OpKind::Tuple, &[p]); // start=0, len=1

        let x = a.push_var(0);
        let y = a.push_var(1);
        let i = a.push_var(4);
        let root = a.push_nary(OpKind::Tuple, &[x, y, i]); // start=1, len=3

        // Any rebuild pass runs every reachable node through `copy_node` for
        // its non-matching arms; `expand_transcendentals` is the simplest
        // public one and this arena has nothing for it to actually lower.
        let new_root = expand_transcendentals(&mut a, root);
        let ExprNode::Nary(OpKind::Tuple, start, len) = a.node(new_root) else {
            panic!("expected a rebuilt Tuple, got {:?}", a.node(new_root));
        };
        let children = a.nary_children_slice(*start, *len);
        assert_eq!(children.len(), 3, "wrong slice length");
        for (child, expected_var) in children.iter().zip([0u8, 1, 4]) {
            assert!(
                matches!(a.node(*child), ExprNode::Var(v) if *v == expected_var),
                "child {child:?} should be Var({expected_var})"
            );
        }
    }

    #[test]
    fn lower_dwrt_refuses_a_malformed_dwrt_shape() {
        // `Dwrt` is only well-formed as `Binary(expr, var)`; any other arity
        // is a malformed node the pass must refuse outright, not silently
        // reinterpret.
        const MALFORMED: &str = "lower_dwrt: malformed Dwrt node (must be Binary(expr, var))";

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let root = a.push_unary(OpKind::Dwrt, x);
        match lower_dwrt_owned(&a, root) {
            Err(msg) => assert_eq!(msg, MALFORMED),
            Ok(_) => panic!("expected {MALFORMED:?}"),
        }

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let z = a.push_const(0.0);
        let root = a.push_ternary(OpKind::Dwrt, x, y, z);
        match lower_dwrt_owned(&a, root) {
            Err(msg) => assert_eq!(msg, MALFORMED),
            Ok(_) => panic!("expected {MALFORMED:?}"),
        }

        // `Nary` is its own alternative in the malformed-shape matcher, and
        // `push_nary` can build one — so without this case, removing that
        // alternative would leave a malformed `Dwrt` reachable while the unary
        // and ternary assertions above still passed.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let root = a.push_nary(OpKind::Dwrt, &[x, y]);
        match lower_dwrt_owned(&a, root) {
            Err(msg) => assert_eq!(msg, MALFORMED),
            Ok(_) => panic!("expected {MALFORMED:?}"),
        }
    }
}

// ─────────────────────────── Passes as optimizers ────────────────────────────
//
// The two passes the runtime tier runs before saturation, as
// [`Optimize`](crate::optimize::Optimize) values so the tier can spell its
// pipeline as a composition instead of three hand-sequenced calls whose order
// only a comment enforces.
//
// Each reports `Unchanged` where its `_owned` wrapper would have cloned the
// arena to say "nothing to do" — the clone was pure waste, and the type now
// has a way to decline it.

use crate::optimize::{Optimize, Rewritten};

/// Replace every `Ref` with its referent.
///
/// First in every pipeline, because a reference is a name and every step
/// after this one reads structure: you cannot differentiate a name, unroll
/// one, or price one.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpandRefs;

impl Optimize for ExpandRefs {
    fn optimize(&mut self, arena: &ExprArena, root: ExprId) -> Rewritten {
        if !arena.nodes().any(|n| matches!(n, ExprNode::Ref(_))) {
            return Rewritten::Unchanged;
        }
        let mut owned = arena.clone();
        let new_root = expand_refs(&mut owned, root);
        Rewritten::Changed(owned, new_root)
    }
}

/// Resolve `Dwrt` (symbolic differentiation) into ordinary arithmetic.
///
/// Runs BEFORE saturation, and the order matters: differentiation manufactures
/// constants — the winding kernels' `d = X − f(Y)` gives `DX(d) = 1` and, for
/// a straight edge, a constant `DY(d)`, making the whole gradient magnitude
/// `√(DX²+DY²)` a compile-time number — and constant folding can only cascade
/// over constants that exist by the time it runs. Lowering after the e-graph
/// leaves those folds permanently on the table, because nothing folds
/// post-extraction.
///
/// Declines on a genuinely non-differentiable op, which stops the pipeline:
/// the term compiles unoptimized and the compile entry's own `lower_dwrt`
/// reports the same error loudly, at the layer that can name it.
#[derive(Clone, Copy, Debug, Default)]
pub struct LowerDwrt;

impl Optimize for LowerDwrt {
    fn optimize(&mut self, arena: &ExprArena, root: ExprId) -> Rewritten {
        if !arena.nodes().any(|n| {
            matches!(
                n,
                ExprNode::Unary(OpKind::Dwrt, _)
                    | ExprNode::Binary(OpKind::Dwrt, _, _)
                    | ExprNode::Ternary(OpKind::Dwrt, _, _, _)
                    | ExprNode::Nary(OpKind::Dwrt, _, _)
            )
        }) {
            return Rewritten::Unchanged;
        }
        let mut owned = arena.clone();
        match lower_dwrt(&mut owned, root) {
            Ok(new_root) => Rewritten::Changed(owned, new_root),
            Err(_) => Rewritten::Declined,
        }
    }
}

/// Unroll every `Reduce` into its terms.
///
/// The extents are static, so the binder disappears into N terms sharing their
/// index-invariant subtrees, and what saturation then sees is binder-free
/// arithmetic it can CSE and fold across those terms — rather than rewriting
/// under a binder.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpandReduce;

impl Optimize for ExpandReduce {
    fn optimize(&mut self, arena: &ExprArena, root: ExprId) -> Rewritten {
        if !arena.nodes().any(|n| matches!(n, ExprNode::Reduce { .. })) {
            return Rewritten::Unchanged;
        }
        let mut owned = arena.clone();
        let new_root = expand_reduce(&mut owned, root);
        Rewritten::Changed(owned, new_root)
    }
}

#[cfg(test)]
mod ref_expansion_tests {
    use super::*;
    use crate::kernel::Kernel;
    use crate::key::canonical;
    use crate::optimize::Rewritten;
    use crate::store::KernelStore;

    /// `√((X − 1.5)² + Y²) − 0.75` — arithmetic with shared subterms, so a
    /// splice that broke DAG sharing would show up in the node count.
    fn circle() -> Kernel {
        let dx = Kernel::x().sub(&Kernel::constant(1.5));
        let dy = Kernel::y();
        dx.mul(&dx)
            .add(&dy.mul(&dy))
            .sqrt()
            .sub(&Kernel::constant(0.75))
    }

    /// Every pass has an identity fast-path, and this one carries the
    /// determinism of the runtime pipeline: a tier that gained a step must
    /// emit the same kernel it did before for every kernel with no reference
    /// in it, which is every kernel production builds today.
    #[test]
    fn expansion_is_an_identity_when_nothing_is_named() {
        let k = circle();
        let (arena, root) = k.parts();
        let (out, out_root) = expand_refs_owned(arena, root);
        assert_eq!(out_root, root, "the root cannot move");
        assert_eq!(out.len(), arena.len(), "no node may be added or dropped");
        assert_eq!(canonical(&out, out_root).key, canonical(arena, root).key);
        assert!(matches!(
            ExpandRefs.optimize(arena, root),
            Rewritten::Unchanged
        ));
    }

    /// `lower_dwrt` on its own refuses a reference rather than inventing a
    /// derivative for a name.
    #[test]
    fn differentiating_a_reference_directly_is_refused() {
        let named = Kernel::x().mul(&Kernel::x()).by_ref().dx();
        let (arena, root) = named.parts();
        match lower_dwrt_owned(arena, root) {
            Err(msg) => assert!(
                msg.contains("cannot differentiate a Ref"),
                "unexpected message: {msg}"
            ),
            Ok(_) => panic!("lower_dwrt must refuse a Ref"),
        }
    }

    /// A key that names nothing is a corrupt graph, reported where it can be
    /// named rather than expanded into whatever happened to be at that slot.
    #[test]
    #[should_panic(expected = "names no interned kernel")]
    fn an_unresolvable_key_is_refused() {
        // A key nothing interned: `resolve` says so, and expansion cannot
        // proceed on a name with no referent.
        let never = Kernel::x().add(&Kernel::constant(3.0e-28));
        let (never_arena, never_root) = never.parts();
        let orphan = crate::key::KernelKey::of(never_arena, never_root);
        assert!(KernelStore::resolve(orphan).is_none(), "must be unknown");
        let mut a = ExprArena::new();
        let root = a.push_ref(orphan);
        let _refused = expand_refs_owned(&a, root);
    }

    /// An *open* term — a `Kernel::over` body still holding its binder's
    /// placeholder — has no identity to name it by: the binder's rename
    /// cannot reach through a name, so expansion would put back an index
    /// nothing binds.
    #[test]
    #[should_panic(expected = "an open term has no identity")]
    fn naming_an_open_term_is_refused() {
        let _refused = Kernel::sum_over(3, |i| i.by_ref());
    }
}
