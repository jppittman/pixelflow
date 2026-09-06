//! IR-to-IR transforms: legalization.
//!
//! Four passes, each `(arena, root) -> (arena, root)`, each turning ops no
//! backend can emit into ops every backend can:
//!
//! | pass | consumes | produces |
//! |---|---|---|
//! | [`lower_dwrt`] | `Dwrt` | arithmetic, and *re-introduces* transcendentals |
//! | [`expand_reduce`] | `Reduce` | the combiner applied over unrolled copies |
//! | [`expand_gather`] | `Gather` | index arithmetic + `RawGather` |
//! | [`expand_transcendentals`] | `Sin`..`Pow` | arithmetic + bit-manip atoms |
//!
//! The order in that table is the order they must run: differentiating a `sin`
//! produces a `cos`, so `lower_dwrt` has to go before the pass that expands
//! them. Every pass is idempotent and has an identity fast-path, so running
//! one that has nothing to do is free.
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
use crate::kind::OpKind;
use crate::variance::Variance;
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
    // `lower_dwrt` first: differentiating a `sin` manufactures a `cos`, so it
    // has to precede the pass that expands them.
    let (arena, root) = lower_dwrt_owned(arena, root)?;
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
    let old_len = arena.nodes_raw().len();
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
                    None => copy_node(arena, &node, &m),
                };
                id_map[id.0 as usize] = Some(new_id);
            }
        }
    }

    Ok(id_map[root.0 as usize].expect("root lowered"))
}

/// Structural copy of `node` into `arena` with its children remapped by `m`.
/// The default action for any node a lowering hook does not replace.
fn copy_node(arena: &mut ExprArena, node: &ExprNode, m: &dyn Fn(ExprId) -> ExprId) -> ExprId {
    match node {
        ExprNode::Var(i) => arena.push_var(*i),
        ExprNode::Const(v) => arena.push_const(*v),
        ExprNode::Param(i) => arena.push_param(*i),
        // Same arena, so the buffer table (and ids) stay valid.
        ExprNode::Buffer(b) => arena.push_buffer(*b),
        ExprNode::Unary(op, a) => arena.push_unary(*op, m(*a)),
        ExprNode::Binary(op, a, b) => arena.push_binary(*op, m(*a), m(*b)),
        ExprNode::Ternary(op, a, b, c) => arena.push_ternary(*op, m(*a), m(*b), m(*c)),
        ExprNode::Nary(op, start, len) => {
            let (s, l) = (*start as usize, *len as usize);
            let children: Vec<ExprId> = arena.nary_children_raw()[s..s + l].to_vec();
            let mapped: Vec<ExprId> = children.into_iter().map(&m).collect();
            arena.push_nary(*op, &mapped)
        }
    }
}

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
    if !arena.nodes_raw().iter().any(|n| match n {
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
        .nodes_raw()
        .iter()
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
        ExprNode::Nary(OpKind::Reduce, start, len) => {
            let (s, l) = (*start as usize, *len as usize);
            debug_assert_eq!(l, 4, "Reduce has 4 children");
            let ch: [ExprId; 4] = {
                let raw = &arena.nary_children_raw()[s..s + l];
                [raw[0], raw[1], raw[2], raw[3]]
            };
            // Children are already lowered; read the (lowered) Const metadata
            // and unroll over the lowered body.
            Some(unroll_reduce(arena, m(ch[0]), m(ch[1]), m(ch[2]), m(ch[3])))
        }
        _ => None,
    })
}

/// Owned wrapper mirroring [`expand_transcendentals_owned`]: identity fast-path
/// when the arena has no `Reduce`, otherwise clone-and-lower.
#[must_use]
pub fn expand_reduce_owned(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    if !arena
        .nodes_raw()
        .iter()
        .any(|n| matches!(n, ExprNode::Nary(OpKind::Reduce, _, _)))
    {
        return (arena.clone(), root);
    }
    let mut owned = arena.clone();
    let new_root = expand_reduce(&mut owned, root);
    (owned, new_root)
}

/// Build the unrolled accumulation for one reduction whose children are already
/// lowered. Reads `combiner`/`var`/`extent` from their `Const` nodes, then folds
/// `extent` substituted copies of `body` under the combiner monoid.
fn unroll_reduce(
    arena: &mut ExprArena,
    combiner: ExprId,
    var: ExprId,
    extent: ExprId,
    body: ExprId,
) -> ExprId {
    let combiner_op = OpKind::from_index(const_val(arena, combiner, "reduce combiner") as usize)
        .expect("reduce combiner must be a valid OpKind index");
    let var_idx = const_val(arena, var, "reduce var") as u8;
    let n = const_val(arena, extent, "reduce extent") as usize;

    // Empty domain folds to the monoid identity.
    if n == 0 {
        let id = combiner_op
            .monoid_identity()
            .expect("reduce combiner is a monoid");
        return arena.push_const(id);
    }

    // Which of the body's nodes actually vary with the index. Everything else
    // is shared across all N terms rather than copied into each of them: the
    // rewrite `⊕_i (f(i) · c) = c · ⊕_i f(i)` obtained by not duplicating `c`
    // in the first place. Computed once here, before the substitutions start
    // appending; every node reachable from `body` predates that point, so the
    // table covers each id the substitution asks about.
    let variance = crate::variance::compute_arena_variance(arena);

    // acc = body[var:=0]; then acc = combiner(acc, body[var:=k]) for k in 1..N.
    let term = |arena: &mut ExprArena, k: usize| {
        Substitution::new(arena, var_idx, k as f32, &variance).apply(arena, body)
    };
    let mut acc = term(arena, 0);
    for k in 1..n {
        let next = term(arena, k);
        acc = arena.push_binary(combiner_op, acc, next);
    }
    acc
}

/// Read the value of a `Const` node (reduction metadata).
fn const_val(arena: &ExprArena, id: ExprId, what: &str) -> f32 {
    match arena.node(id) {
        ExprNode::Const(v) => *v,
        other => panic!("{what} must be a Const, got {other:?}"),
    }
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
    memo: Vec<Option<ExprId>>,
}

impl<'a> Substitution<'a> {
    fn new(arena: &ExprArena, var: u8, value: f32, variance: &'a [Variance]) -> Self {
        Self {
            var,
            value,
            variance,
            memo: alloc::vec![None; arena.nodes_raw().len()],
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
            ExprNode::Nary(op, start, len) => {
                let (s, l) = (start as usize, len as usize);
                let children: Vec<ExprId> = arena.nary_children_raw()[s..s + l].to_vec();
                let mapped: Vec<ExprId> = children
                    .into_iter()
                    .map(|ch| self.apply(arena, ch))
                    .collect();
                arena.push_nary(op, &mapped)
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
/// Errors loudly on any op with no derivative rule (bound-memory reads,
/// integer/bit ops, reductions) rather than silently miscompiling.
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
    if !arena.nodes_raw().iter().any(|n| {
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
fn differentiate(arena: &mut ExprArena, expr: ExprId, var: u8) -> Result<ExprId, &'static str> {
    let entry_len = arena.nodes_raw().len();

    // Pass 1: mark derivative-needed nodes.
    let mut need = alloc::vec![false; entry_len];
    let mut stack: Vec<ExprId> = alloc::vec![expr];
    while let Some(id) = stack.pop() {
        if core::mem::replace(&mut need[id.0 as usize], true) {
            continue;
        }
        push_deriv_children(arena.node(id), &mut stack);
    }

    // Pass 2: bottom-up compute in topological (id) order.
    let mut memo: Vec<Option<ExprId>> = alloc::vec![None; entry_len];
    for idx in 0..entry_len {
        if !need[idx] {
            continue;
        }
        let d = diff_node(arena, ExprId(idx as u32), var, &memo)?;
        memo[idx] = Some(d);
    }
    Ok(memo[expr.0 as usize].expect("derivative of the root was computed"))
}

/// Which children's derivatives the rule for `node` consumes. Must stay in
/// lockstep with [`diff_node`]: a child pushed here is differentiated eagerly;
/// a child omitted here must not be read from the memo there. Ops with no
/// rule push nothing — [`diff_node`] raises the error for the node itself.
fn push_deriv_children(node: &ExprNode, stack: &mut Vec<ExprId>) {
    match *node {
        ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Param(_) | ExprNode::Buffer(_) => {}
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
    }
}

fn diff_node(
    arena: &mut ExprArena,
    id: ExprId,
    var: u8,
    memo: &[Option<ExprId>],
) -> Result<ExprId, &'static str> {
    match arena.node(id).clone() {
        ExprNode::Var(i) => Ok(arena.push_const(if i == var { 1.0 } else { 0.0 })),
        // Constants and scalar params (baked before evaluation) are
        // coordinate-independent.
        ExprNode::Const(_) | ExprNode::Param(_) => Ok(arena.push_const(0.0)),
        ExprNode::Buffer(_) => Err("lower_dwrt: cannot differentiate a bound-memory read"),

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
            OpKind::RawGather => Err("lower_dwrt: cannot differentiate a bound-memory read"),
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
            OpKind::Gather => Err("lower_dwrt: cannot differentiate a bound-memory read"),
            _ => Err("lower_dwrt: no derivative rule for this ternary op"),
        },

        ExprNode::Nary(_, _, _) => {
            Err("lower_dwrt: cannot differentiate an Nary op (Reduce/Tuple)")
        }
    }
}

/// Read a child's already-computed derivative. Pass 1 marks exactly the
/// children each rule consumes and pass 2 runs bottom-up, so the entry is
/// always populated when the parent's rule fires.
fn dchild(memo: &[Option<ExprId>], child: ExprId) -> ExprId {
    memo[child.0 as usize].expect("child derivative marked and computed before parent")
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
/// Mirrors the runtime `SimdOps` version: reduce to a ratio in [-1,1] (swapping
/// y/x when |y|>|x|), a degree-7 odd polynomial for atan on that interval, then
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
fn expand_exp2(arena: &mut ExprArena, x: ExprId) -> ExprId {
    // Clamp to a safe exponent range to avoid int overflow / inf.
    let lo = arena.push_const(-126.0);
    let hi = arena.push_const(126.0);
    let x = arena.push_binary(OpKind::Max, x, lo);
    let x = arena.push_binary(OpKind::Min, x, hi);

    // xi = floor(x), xf = x - xi
    let xi = arena.push_unary(OpKind::Floor, x);
    let xf = arena.push_binary(OpKind::Sub, x, xi);

    // 2^xf ≈ Horner(c5..c0) at xf  (minimax coefficients).
    let c0 = arena.push_const(1.0);
    let c1 = arena.push_const(core::f32::consts::LN_2);
    let c2 = arena.push_const(0.240_226_5);
    let c3 = arena.push_const(0.055_504_11);
    let c4 = arena.push_const(0.009_618_129);
    let c5 = arena.push_const(0.001_333_355_8);
    let p = horner_step(arena, c5, xf, c4);
    let p = horner_step(arena, p, xf, c3);
    let p = horner_step(arena, p, xf, c2);
    let p = horner_step(arena, p, xf, c1);
    let p = horner_step(arena, p, xf, c0);

    // 2^xi = bitcast((int(xi) + 127) << 23).
    let xi_int = arena.push_unary(OpKind::TruncToInt, xi);
    let bias = arena.push_const(f32::from_bits(127)); // integer 127 as lane bits
    let biased = arena.push_binary(OpKind::IAdd, xi_int, bias);
    // Shift amount is read by value (`v as u32 as u8`), so it is a plain 23.0.
    let shift = arena.push_const(23.0);
    let pow2i = arena.push_binary(OpKind::Shl, biased, shift); // bitcast result

    // 2^x = 2^xf · 2^xi
    arena.push_binary(OpKind::Mul, p, pow2i)
}

/// `log2(x)` as a primitive subgraph (x > 0).
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
    let c8 = arena.push_const(7.037_683_6e-2);
    let c7 = arena.push_const(-1.151_461e-1);
    let c6 = arena.push_const(1.167_699_9e-1);
    let c5 = arena.push_const(-1.242_014_1e-1);
    let c4 = arena.push_const(1.424_932_3e-1);
    let c3 = arena.push_const(-1.666_805_8e-1);
    let c2 = arena.push_const(2.000_071_5e-1);
    let c1 = arena.push_const(-2.499_999_4e-1);
    let c0 = arena.push_const(3.333_333e-1);
    let p = horner_step(arena, c8, t, c7);
    let p = horner_step(arena, p, t, c6);
    let p = horner_step(arena, p, t, c5);
    let p = horner_step(arena, p, t, c4);
    let p = horner_step(arena, p, t, c3);
    let p = horner_step(arena, p, t, c2);
    let p = horner_step(arena, p, t, c1);
    let p = horner_step(arena, p, t, c0);

    // y = t³·P(t) − t²/2, so ln(1+t) = t + y.
    let t2 = arena.push_binary(OpKind::Mul, t, t);
    let t3 = arena.push_binary(OpKind::Mul, t2, t);
    let t3p = arena.push_binary(OpKind::Mul, t3, p);
    let half_t2 = arena.push_binary(OpKind::Mul, t2, half);
    let y = arena.push_binary(OpKind::Sub, t3p, half_t2);

    // log2(m) = (t + y)·log2(e), with log2(e) split as 1 + LOG2EA and the
    // pieces summed smallest-first (Cephes ordering) to keep full precision:
    // e + t + y + y·LOG2EA + t·LOG2EA.
    let log2ea = arena.push_const(0.442_695_04); // log2(e) − 1
    let y_ea = arena.push_binary(OpKind::Mul, y, log2ea);
    let t_ea = arena.push_binary(OpKind::Mul, t, log2ea);
    let z = arena.push_binary(OpKind::Add, y_ea, t_ea);
    let z = arena.push_binary(OpKind::Add, z, y);
    let z = arena.push_binary(OpKind::Add, z, t);
    arena.push_binary(OpKind::Add, z, e)
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

/// `acc·x + add` as `Add(Mul(acc, x), add)` — two nodes, not one `MulAdd`.
///
/// Two tempting justifications for it are simply untrue, and worth stating so
/// they are not assumed: the jet path *does* have a `MulAdd` rule (`diff_node`,
/// `ExprNode::Ternary`), and nothing re-fuses these afterwards — [`legalize`] is
/// the last thing to touch the arena, and the only thing that becomes an FMA
/// instruction is an `OpKind::MulAdd` node already in it. So the Horner chains
/// here reach the emitter unfused: 5 mul + 5 add for `sin`, where 5 `MulAdd`s
/// would do.
///
/// The real reason to keep it is parity. Unfused mul+add rounds twice on every
/// target, so the `eval_scalar` oracle and every backend agree exactly. `MulAdd`
/// rounds *once* where FMA exists and twice where it does not (CLAUDE.md,
/// "Floating point at the edges"), which would reintroduce a tier divergence —
/// small, but this expansion is precisely where a shared-definition disagreement
/// between oracle and JIT went unnoticed for so long. Fusing is a real ~5-op win
/// per `sin` available for the taking; it should be taken deliberately, with the
/// parity tests updated to expect a 1-ulp tier difference, not as a side effect.
fn horner_step(arena: &mut ExprArena, acc: ExprId, x: ExprId, add: ExprId) -> ExprId {
    let prod = arena.push_binary(OpKind::Mul, acc, x);
    arena.push_binary(OpKind::Add, prod, add)
}

#[cfg(test)]
mod dwrt_tests {
    use super::*;
    use crate::binding::BindingTable;
    use crate::eval::eval_scalar;

    /// Wrap `expr` in `Dwrt(expr, var)`, run [`lower_dwrt`], and assert no
    /// `Dwrt` is reachable from the new root (the rebuild leaves the original
    /// `Dwrt` behind as a dead node, which the scheduler's reachability filter
    /// drops).
    fn lowered_derivative(arena: &ExprArena, expr: ExprId, var: u8) -> (ExprArena, ExprId) {
        let mut a = arena.clone();
        let v = a.push_const(var as f32);
        let root = a.push_binary(OpKind::Dwrt, expr, v);
        let (out, out_root) = lower_dwrt_owned(&a, root).expect("lower_dwrt");
        assert!(
            !reachable_dwrt(&out, out_root),
            "lowered derivative still contains a reachable Dwrt",
        );
        (out, out_root)
    }

    fn reachable_dwrt(arena: &ExprArena, root: ExprId) -> bool {
        let mut seen = alloc::vec![false; arena.nodes_raw().len()];
        let mut stack = alloc::vec![root];
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            if matches!(
                arena.node(id),
                ExprNode::Unary(OpKind::Dwrt, _)
                    | ExprNode::Binary(OpKind::Dwrt, _, _)
                    | ExprNode::Ternary(OpKind::Dwrt, _, _, _)
            ) {
                return true;
            }
            stack.extend(arena.children(id));
        }
        false
    }

    fn eval(arena: &ExprArena, root: ExprId, vars: &[f32; 4]) -> f32 {
        eval_scalar(arena, root, vars, &BindingTable::empty())
    }

    fn assert_close(got: f32, want: f32, pt: &[f32; 4]) {
        assert_close_rel(got, want, pt, 1e-3);
    }

    /// Relative tolerance the caller chooses.
    ///
    /// Derivatives of transcendentals need a looser bound than exact
    /// arithmetic: the interpreter evaluates the language's own polynomial
    /// expansion (not the host libm — see `eval_scalar`), so comparing against
    /// `f32::cos` measures the derivative rule *and* the ~4e-3 error of the
    /// 4-term Chebyshev approximation. That error is the language's actual
    /// answer, and pinning it here is the point: the tolerance documents the
    /// approximation instead of hiding it behind an exact host function.
    fn assert_close_rel(got: f32, want: f32, pt: &[f32; 4], rel: f32) {
        let tol = rel * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "at {pt:?}: got {got}, want {want} (tol {tol})"
        );
    }

    #[test]
    fn differentiate_a_variable_to_one_for_itself_and_zero_for_the_others() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let (out, root) = lowered_derivative(&a, x, 0);
        assert_close(
            eval(&out, root, &[3.0, 5.0, 0.0, 0.0]),
            1.0,
            &[3.0, 5.0, 0.0, 0.0],
        );

        let mut a = ExprArena::new();
        let y = a.push_var(1);
        let (out, root) = lowered_derivative(&a, y, 0);
        assert_close(
            eval(&out, root, &[3.0, 5.0, 0.0, 0.0]),
            0.0,
            &[3.0, 5.0, 0.0, 0.0],
        );
    }

    #[test]
    fn compose_the_sqrt_rule_with_the_chain_rule_over_a_sum_of_squares() {
        // d/dx √(x² + y²) = x / √(x² + y²) — the font-SDF core.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let x2 = a.push_binary(OpKind::Mul, x, x);
        let y2 = a.push_binary(OpKind::Mul, y, y);
        let sum = a.push_binary(OpKind::Add, x2, y2);
        let e = a.push_unary(OpKind::Sqrt, sum);
        let (out, root) = lowered_derivative(&a, e, 0);
        for p in &[
            [3.0f32, 4.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [-2.0, 5.0, 0.0, 0.0],
        ] {
            let want = p[0] / (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert_close(eval(&out, root, p), want, p);
        }
    }

    #[test]
    fn take_the_derivative_of_whichever_branch_min_and_max_select() {
        // d/dx min(x·2, y·3) is 2 where x·2 < y·3, else 0 (and dually for max).
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let two = a.push_const(2.0);
        let three = a.push_const(3.0);
        let x2 = a.push_binary(OpKind::Mul, x, two);
        let y3 = a.push_binary(OpKind::Mul, y, three);
        let e = a.push_binary(OpKind::Min, x2, y3);
        let (out, root) = lowered_derivative(&a, e, 0);
        assert_close(
            eval(&out, root, &[1.0, 5.0, 0.0, 0.0]),
            2.0,
            &[1.0, 5.0, 0.0, 0.0],
        );
        assert_close(
            eval(&out, root, &[9.0, 1.0, 0.0, 0.0]),
            0.0,
            &[9.0, 1.0, 0.0, 0.0],
        );

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let two = a.push_const(2.0);
        let three = a.push_const(3.0);
        let x2 = a.push_binary(OpKind::Mul, x, two);
        let y3 = a.push_binary(OpKind::Mul, y, three);
        let e = a.push_binary(OpKind::Max, x2, y3);
        let (out, root) = lowered_derivative(&a, e, 0);
        assert_close(
            eval(&out, root, &[9.0, 1.0, 0.0, 0.0]),
            2.0,
            &[9.0, 1.0, 0.0, 0.0],
        );
        assert_close(
            eval(&out, root, &[1.0, 5.0, 0.0, 0.0]),
            0.0,
            &[1.0, 5.0, 0.0, 0.0],
        );
    }

    #[test]
    fn blend_the_branches_derivatives_by_the_same_mask_select_used() {
        // d/dx select(y > 0, x·x, x·5) = 2x above the axis, 5 below.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let zero = a.push_const(0.0);
        let five = a.push_const(5.0);
        let mask = a.push_binary(OpKind::Gt, y, zero);
        let xx = a.push_binary(OpKind::Mul, x, x);
        let x5 = a.push_binary(OpKind::Mul, x, five);
        let e = a.push_ternary(OpKind::Select, mask, xx, x5);
        let (out, root) = lowered_derivative(&a, e, 0);
        assert_close(
            eval(&out, root, &[3.0, 1.0, 0.0, 0.0]),
            6.0,
            &[3.0, 1.0, 0.0, 0.0],
        );
        assert_close(
            eval(&out, root, &[3.0, -1.0, 0.0, 0.0]),
            5.0,
            &[3.0, -1.0, 0.0, 0.0],
        );
    }

    #[test]
    fn give_a_clamped_expression_zero_derivative_outside_its_bounds() {
        // d/dx clamp(x·x, 0, 10): 2x inside, 0 once saturated. `clamp` is
        // library, so this is the min/max composition and the derivative comes
        // from the min/max rules — no clamp-specific rule exists any more.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let zero = a.push_const(0.0);
        let ten = a.push_const(10.0);
        let xx = a.push_binary(OpKind::Mul, x, x);
        let floored = a.push_binary(OpKind::Max, xx, zero);
        let e = a.push_binary(OpKind::Min, floored, ten);
        let (out, root) = lowered_derivative(&a, e, 0);
        assert_close(
            eval(&out, root, &[2.0, 0.0, 0.0, 0.0]),
            4.0,
            &[2.0, 0.0, 0.0, 0.0],
        );
        assert_close(
            eval(&out, root, &[5.0, 0.0, 0.0, 0.0]),
            0.0,
            &[5.0, 0.0, 0.0, 0.0],
        );
    }

    #[test]
    fn differentiate_mul_add_by_the_product_rule_plus_the_addends_derivative() {
        // d/dx (x·y + x) = y + 1.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let e = a.push_ternary(OpKind::MulAdd, x, y, x);
        let (out, root) = lowered_derivative(&a, e, 0);
        for p in &[[2.0f32, 3.0, 0.0, 0.0], [-1.0, 7.0, 0.0, 0.0]] {
            assert_close(eval(&out, root, p), p[1] + 1.0, p);
        }
    }

    #[test]
    fn differentiate_sin_exp_and_ln_to_their_own_rules_under_composition() {
        // d/dx sin(x) = cos(x); d/dx exp(x·x) = 2x·exp(x²); d/dx ln(x) = 1/x.
        //
        // The expected value is built as an arena expression and evaluated the
        // same way, NOT taken from the host libm. That isolates what this test
        // is for: the derivative *rule*. `cos` in this language is the
        // expansion `sin(x + π/2)`, whose 4-term polynomial degrades as the
        // shifted argument approaches π — comparing against `f32::cos` would
        // charge that approximation error to the chain rule and force a
        // tolerance loose enough to hide a real rule bug. Polynomial accuracy
        // versus libm is a separate concern, measured in the emit tests.
        let pts = [[0.7f32, 0.0, 0.0, 0.0], [1.3, 0.0, 0.0, 0.0]];

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let e = a.push_unary(OpKind::Sin, x);
        let expected_cos = a.push_unary(OpKind::Cos, x);
        let (out, root) = lowered_derivative(&a, e, 0);
        for p in &pts {
            let want = eval(&a, expected_cos, p);
            assert_close(eval(&out, root, p), want, p);
        }

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let xx = a.push_binary(OpKind::Mul, x, x);
        let e = a.push_unary(OpKind::Exp, xx);
        // 2x·exp(x²), in the language.
        let two = a.push_const(2.0);
        let two_x = a.push_binary(OpKind::Mul, two, x);
        let exp_xx = a.push_unary(OpKind::Exp, xx);
        let expected = a.push_binary(OpKind::Mul, two_x, exp_xx);
        let (out, root) = lowered_derivative(&a, e, 0);
        for p in &pts {
            let want = eval(&a, expected, p);
            assert_close(eval(&out, root, p), want, p);
        }

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let e = a.push_unary(OpKind::Ln, x);
        let (out, root) = lowered_derivative(&a, e, 0);
        for p in &pts {
            assert_close(eval(&out, root, p), 1.0 / p[0], p);
        }
    }

    #[test]
    fn nested_dwrt_is_second_derivative() {
        // d²/dx² (x·x·x) = 6x, via Dwrt(Dwrt(x³, 0), 0).
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let xx = a.push_binary(OpKind::Mul, x, x);
        let xxx = a.push_binary(OpKind::Mul, xx, x);
        let v0 = a.push_const(0.0);
        let d1 = a.push_binary(OpKind::Dwrt, xxx, v0);
        let root = a.push_binary(OpKind::Dwrt, d1, v0);
        let (out, out_root) = lower_dwrt_owned(&a, root).expect("lower_dwrt");
        for p in &[[2.0f32, 0.0, 0.0, 0.0], [-1.5, 0.0, 0.0, 0.0]] {
            assert_close(eval(&out, out_root, p), 6.0 * p[0], p);
        }
    }

    #[test]
    fn shared_subgraph_differentiates_once() {
        // A DAG: s = x·y used twice. The derivative must stay a DAG (no
        // exponential blowup) and be correct: d/dx (s·s) = 2·s·y.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let s = a.push_binary(OpKind::Mul, x, y);
        let e = a.push_binary(OpKind::Mul, s, s);
        let (out, root) = lowered_derivative(&a, e, 0);
        let p = [3.0f32, 2.0, 0.0, 0.0];
        assert_close(eval(&out, root, &p), 2.0 * (p[0] * p[1]) * p[1], &p);
    }

    #[test]
    fn no_dwrt_is_identity() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let e = a.push_binary(OpKind::Add, x, y);
        let (out, root) = lower_dwrt_owned(&a, e).expect("lower_dwrt");
        assert_eq!(out.nodes_raw().len(), a.nodes_raw().len());
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
        assert!(out.nodes_raw().len() > a.nodes_raw().len());
        assert!((out_root.0 as usize) < out.nodes_raw().len());
    }

    #[test]
    fn unsupported_op_errors_loudly() {
        // Differentiating a Reduce has no rule: the pass must refuse.
        let mut a = ExprArena::new();
        let combiner = a.push_const(OpKind::Add.index() as f32);
        let rvar = a.push_const(0.0);
        let extent = a.push_const(4.0);
        let body = a.push_var(4);
        let red = a.push_nary(OpKind::Reduce, &[combiner, rvar, extent, body]);
        let v0 = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, red, v0);
        assert!(lower_dwrt_owned(&a, root).is_err());
    }

    #[test]
    fn flip_the_sign_for_neg_and_square_the_denominator_for_recip() {
        // d/dx -(x·x) = -2x.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let xx = a.push_binary(OpKind::Mul, x, x);
        let e = a.push_unary(OpKind::Neg, xx);
        let (out, root) = lowered_derivative(&a, e, 0);
        for p in &[[3.0f32, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]] {
            assert_close(eval(&out, root, p), -2.0 * p[0], p);
        }

        // d/dx (1/x) = -1/x².
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let e = a.push_unary(OpKind::Recip, x);
        let (out, root) = lowered_derivative(&a, e, 0);
        for p in &[[2.0f32, 0.0, 0.0, 0.0], [-4.0, 0.0, 0.0, 0.0]] {
            assert_close(eval(&out, root, p), -1.0 / (p[0] * p[0]), p);
        }
    }

    #[test]
    fn differentiate_abs_to_the_sign_of_its_operand() {
        // d/dx |x| = x/|x| — +1 above zero, -1 below.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let e = a.push_unary(OpKind::Abs, x);
        let (out, root) = lowered_derivative(&a, e, 0);
        assert_close(
            eval(&out, root, &[3.0, 0.0, 0.0, 0.0]),
            1.0,
            &[3.0, 0.0, 0.0, 0.0],
        );
        assert_close(
            eval(&out, root, &[-3.0, 0.0, 0.0, 0.0]),
            -1.0,
            &[-3.0, 0.0, 0.0, 0.0],
        );
    }

    #[test]
    fn differentiate_rsqrt_to_minus_half_x_to_the_negative_three_halves() {
        // d/dx x^(-1/2) = -0.5 · x^(-3/2).
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let e = a.push_unary(OpKind::Rsqrt, x);
        let (out, root) = lowered_derivative(&a, e, 0);
        for xv in [4.0f32, 9.0] {
            let want = -0.5 * xv.powf(-1.5);
            let p = [xv, 0.0, 0.0, 0.0];
            assert_close(eval(&out, root, &p), want, &p);
        }
    }

    #[test]
    fn match_the_closed_forms_for_the_remaining_trig_and_inverse_trig_rules() {
        // Expected values for the transcendental cases are built as arena
        // expressions and evaluated the same way as the derivative under
        // test, NOT taken from host libm — see `differentiate_sin_exp_and_ln_to_their_own_rules_under_composition` for why:
        // it isolates the chain-rule from the polynomial-approximation error.
        let pt = [0.4f32, 0.0, 0.0, 0.0];

        // d/dx cos(x) = -sin(x).
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let e = a.push_unary(OpKind::Cos, x);
        let sinx = a.push_unary(OpKind::Sin, x);
        let expected = a.push_unary(OpKind::Neg, sinx);
        let (out, root) = lowered_derivative(&a, e, 0);
        let want = eval(&a, expected, &pt);
        assert_close(eval(&out, root, &pt), want, &pt);

        // d/dx tan(x) = 1/cos²(x).
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let e = a.push_unary(OpKind::Tan, x);
        let cosx = a.push_unary(OpKind::Cos, x);
        let cos2 = a.push_binary(OpKind::Mul, cosx, cosx);
        let one = a.push_const(1.0);
        let expected = a.push_binary(OpKind::Div, one, cos2);
        let (out, root) = lowered_derivative(&a, e, 0);
        let want = eval(&a, expected, &pt);
        assert_close(eval(&out, root, &pt), want, &pt);

        // d/dx asin(x) = 1/√(1-x²); d/dx acos(x) = -that. √ and arithmetic
        // are exact in this interpreter, so a closed form is fine here.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let e = a.push_unary(OpKind::Asin, x);
        let (out, root) = lowered_derivative(&a, e, 0);
        let want = 1.0 / (1.0 - pt[0] * pt[0]).sqrt();
        assert_close(eval(&out, root, &pt), want, &pt);

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let e = a.push_unary(OpKind::Acos, x);
        let (out, root) = lowered_derivative(&a, e, 0);
        assert_close(eval(&out, root, &pt), -want, &pt);

        // d/dx atan(x) = 1/(1+x²) — pure arithmetic, no transcendental in
        // the derivative expression itself.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let e = a.push_unary(OpKind::Atan, x);
        let (out, root) = lowered_derivative(&a, e, 0);
        for xv in [0.5f32, 2.0, -3.0] {
            let p = [xv, 0.0, 0.0, 0.0];
            let want = 1.0 / (1.0 + xv * xv);
            assert_close(eval(&out, root, &p), want, &p);
        }
    }

    #[test]
    fn carry_the_right_constants_in_the_base_two_and_base_ten_exp_and_log_rules() {
        // d/dx 2^x = 2^x · ln2.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let e = a.push_unary(OpKind::Exp2, x);
        let exp2x = a.push_unary(OpKind::Exp2, x);
        let ln2 = a.push_const(core::f32::consts::LN_2);
        let expected = a.push_binary(OpKind::Mul, exp2x, ln2);
        let (out, root) = lowered_derivative(&a, e, 0);
        for xv in [0.3f32, 2.0, -1.0] {
            let p = [xv, 0.0, 0.0, 0.0];
            let want = eval(&a, expected, &p);
            assert_close(eval(&out, root, &p), want, &p);
        }

        // d/dx log2(x) = 1/(x·ln2) — pure arithmetic given ln2 is a constant.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let e = a.push_unary(OpKind::Log2, x);
        let (out, root) = lowered_derivative(&a, e, 0);
        for xv in [0.5f32, 3.0] {
            let p = [xv, 0.0, 0.0, 0.0];
            let want = 1.0 / (xv * core::f32::consts::LN_2);
            assert_close(eval(&out, root, &p), want, &p);
        }

        // d/dx log10(x) = 1/(x·ln10).
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let e = a.push_unary(OpKind::Log10, x);
        let (out, root) = lowered_derivative(&a, e, 0);
        for xv in [0.5f32, 3.0] {
            let p = [xv, 0.0, 0.0, 0.0];
            let want = 1.0 / (xv * core::f32::consts::LN_10);
            assert_close(eval(&out, root, &p), want, &p);
        }
    }

    #[test]
    fn negate_subs_right_derivative_and_follow_the_quotient_rule_for_div() {
        // d/dx (x·x - x) = 2x - 1. Both operands must depend on x: with a
        // constant-in-x right operand `db` is zero, and `da - db` and
        // `da + db` agree — the sign of Sub's right term would go unpinned.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let xx = a.push_binary(OpKind::Mul, x, x);
        let e = a.push_binary(OpKind::Sub, xx, x);
        let (out, root) = lowered_derivative(&a, e, 0);
        let p = [3.0f32, 5.0, 0.0, 0.0];
        assert_close(eval(&out, root, &p), 2.0 * p[0] - 1.0, &p);

        // d/dx (x·x / (x + y)) = (2x(x+y) - x²)/(x+y)² — the full quotient
        // rule. Same reason: an x-independent denominator makes `db` zero and
        // collapses the rule to `da / b`, so the `-a·db` term could be
        // deleted outright and this would still pass.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let xx = a.push_binary(OpKind::Mul, x, x);
        let denom = a.push_binary(OpKind::Add, x, y);
        let e = a.push_binary(OpKind::Div, xx, denom);
        let (out, root) = lowered_derivative(&a, e, 0);
        let p = [3.0f32, 2.0, 0.0, 0.0];
        let (xv, b) = (p[0], p[0] + p[1]);
        assert_close(eval(&out, root, &p), (2.0 * xv * b - xv * xv) / (b * b), &p);
    }

    #[test]
    fn differentiate_atan2_and_pow_through_both_of_their_operands() {
        // d/dX atan2(Y, X) = (X·dY - Y·dX)/(X²+Y²) = -Y/(X²+Y²), since Y does
        // not depend on X. `Atan2`'s children are (y, x), matching `f32::atan2`.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let e = a.push_binary(OpKind::Atan2, y, x);
        let (out, root) = lowered_derivative(&a, e, 0);
        let p = [3.0f32, 4.0, 0.0, 0.0];
        let want = -p[1] / (p[0] * p[0] + p[1] * p[1]);
        assert_close(eval(&out, root, &p), want, &p);

        // Both Atan2 children depending on X, so the `x·dy` half of
        // (x·dy - y·dx)/(x²+y²) is exercised too — with `dy == 0` above, that
        // whole term could be deleted and the assertion would not notice.
        // d/dX atan2(X², X) = (X·2X - X²)/(X⁴ + X²) = X²/(X⁴ + X²).
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let xx = a.push_binary(OpKind::Mul, x, x);
        let e = a.push_binary(OpKind::Atan2, xx, x);
        let (out, root) = lowered_derivative(&a, e, 0);
        let p = [3.0f32, 0.0, 0.0, 0.0];
        let xv = p[0];
        let want = (xv * xv) / (xv * xv * xv * xv + xv * xv);
        assert_close(eval(&out, root, &p), want, &p);

        // d/dx x³ (constant exponent) = 3x², the ordinary power rule falling
        // out of Pow's general f^g·(g'·ln f + g·f'/f) formula.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let three = a.push_const(3.0);
        let e = a.push_binary(OpKind::Pow, x, three);
        let (out, root) = lowered_derivative(&a, e, 0);
        let p = [2.0f32, 0.0, 0.0, 0.0];
        assert_close(eval(&out, root, &p), 3.0 * p[0] * p[0], &p);

        // A constant exponent leaves `dg` zero, so the general rule's
        // `g'·ln(f)` term is unexercised above and could be deleted. With the
        // exponent varying too: d/dx x^x = x^x·(ln x + 1).
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let e = a.push_binary(OpKind::Pow, x, x);
        let (out, root) = lowered_derivative(&a, e, 0);
        let p = [2.0f32, 0.0, 0.0, 0.0];
        let xv = p[0];
        let want = libm::powf(xv, xv) * (libm::logf(xv) + 1.0);
        // Pow expands through exp/ln polynomial fits, so hold this to the
        // same looser relative tolerance the expansions are checked at.
        assert_close_rel(eval(&out, root, &p), want, &p, 3e-2);
    }

    #[test]
    fn multiply_every_unary_rule_by_its_operands_derivative() {
        // Each rule above applies its op to `Var(0)` directly, where the chain
        // rule's `da` factor is exactly 1 — so a rule that dropped or miswired
        // `da` would still pass every one of them. Here each op wraps `x·x`,
        // whose derivative is `2x`, which makes that factor observable.
        //
        // The oracle is the same rule evaluated one level up: `d/du f(u)` at
        // `u = x²`, times `2x`. That deliberately does not re-derive f' by
        // hand — this test is about the chain rule's factor, and reusing the
        // rule for `f'` keeps a polynomial's accuracy out of the comparison
        // exactly as `differentiate_sin_exp_and_ln_to_their_own_rules_under_composition` explains.
        //
        // `x = 0.6` puts `x² = 0.36` inside every domain at once: within
        // [-1, 1] for Asin/Acos, strictly positive for the logs, and nonzero
        // for Recip/Rsqrt.
        const X: f32 = 0.6;
        let outer = [X, 0.0, 0.0, 0.0];
        let inner = [X * X, 0.0, 0.0, 0.0];

        for op in [
            OpKind::Sin,
            OpKind::Cos,
            OpKind::Tan,
            OpKind::Asin,
            OpKind::Acos,
            OpKind::Atan,
            OpKind::Exp,
            OpKind::Exp2,
            OpKind::Ln,
            OpKind::Log2,
            OpKind::Log10,
            OpKind::Sqrt,
            OpKind::Rsqrt,
            OpKind::Recip,
            OpKind::Neg,
            OpKind::Abs,
        ] {
            // d/dx f(x²)
            let mut composed = ExprArena::new();
            let x = composed.push_var(0);
            let xx = composed.push_binary(OpKind::Mul, x, x);
            let e = composed.push_unary(op, xx);
            let (out, root) = lowered_derivative(&composed, e, 0);
            let got = eval(&out, root, &outer);

            // f'(u) at u = x², from the same rule with a unit-derivative
            // operand — the case the tests above already cover.
            let mut bare = ExprArena::new();
            let u = bare.push_var(0);
            let e = bare.push_unary(op, u);
            let (out, root) = lowered_derivative(&bare, e, 0);
            let want = eval(&out, root, &inner) * 2.0 * X;

            // Guard against a vacuous comparison: if `f'(x²)·2x` happened to
            // land on zero, dropping `da` entirely would also produce zero.
            assert!(
                want.abs() > 1e-3,
                "{op:?}: oracle {want} is too near zero at x={X} to distinguish \
                 a present chain-rule factor from a missing one"
            );
            assert_close(got, want, &outer);
        }
    }

    #[test]
    fn differentiate_a_raw_comparison_of_any_kind_to_zero() {
        // A bare comparison (not wrapped in a Select) is a step function:
        // zero derivative, and — unlike an op with no rule at all —
        // `lower_dwrt` must succeed rather than error.
        //
        // All six are separate alternatives in `diff_node`'s and
        // `push_deriv_children`'s grouped matches, so covering only `Lt` would
        // let a dropped or misrouted arm for any of the other five through.
        for op in [
            OpKind::Lt,
            OpKind::Le,
            OpKind::Gt,
            OpKind::Ge,
            OpKind::Eq,
            OpKind::Ne,
        ] {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let e = a.push_binary(op, x, y);
            let (out, root) = lowered_derivative(&a, e, 0);
            // Both orderings and equality, so no arm can pass by accident of
            // the operands it was handed.
            for p in &[
                [1.0f32, 2.0, 0.0, 0.0],
                [2.0f32, 1.0, 0.0, 0.0],
                [1.0f32, 1.0, 0.0, 0.0],
            ] {
                assert_close(eval(&out, root, p), 0.0, p);
            }
        }
    }

    #[test]
    fn transcendentals_evaluate_close_to_host_libm() {
        // No transcendental has a scalar `eval_unary`/`eval_binary` arm (see
        // `kind.rs`) — evaluating one at all requires `expand_transcendentals`
        // to have lowered it to arithmetic first. The derivative-rule tests
        // above build several of these ops as intermediate values, but most
        // of them (Log2, Log10, Ln, Atan, Asin, Acos, Atan2, Pow) only ever
        // appear as pure-arithmetic derivative *results*, never evaluated
        // themselves. This test evaluates each expansion directly across a
        // spread of magnitudes and signs (range reduction and quadrant
        // selection inside the expansions branch on both), checked against
        // host libm, at a tolerance sized per expansion rather than one loose
        // bound for all of them.
        //
        // `LIBM_TOL` is for the expansions that genuinely need it — `exp`,
        // `exp2`, and `atan`, whose fits carry real approximation error by
        // design (`ATAN_MINIMAX` is documented at ~8.7e-5).
        //
        // `TRIG_TOL` is separate because `SIN_CHEB` is six odd coefficients —
        // a degree-11 fit, not the degree-7 one an earlier version of this
        // comment claimed — and the module documents ~1.5e-6 for sin/cos.
        // Measured against libm at exactly the points below, the worst error
        // is 4.2e-7 in this test's own metric, so 3e-2 was roughly 70,000x
        // looser than the implementation: at `x = 0.1`, where
        // `assert_close_rel`'s `max(|want|, 1)` floor makes the bound a plain
        // absolute 0.03, a `sin` that returned 0.13 would have passed. 1e-5
        // keeps ~24x headroom over the measurement and still sits above the
        // documented accuracy, which leaves room for the ISA levels where FMA
        // contraction shifts the last bits, while staying tight enough that a
        // wrong coefficient, sign, or branch cannot hide.
        type Reference = fn(f32) -> f32;

        const LIBM_TOL: f32 = 3e-2;
        const TRIG_TOL: f32 = 1e-5;
        let periodic_pts = [-100.0f32, -7.0, -0.6, 0.1, 0.6, 2.5, 7.0, 100.0];
        let unary: [(OpKind, Reference); 3] = [
            (OpKind::Sin, libm::sinf),
            (OpKind::Cos, libm::cosf),
            (OpKind::Tan, libm::tanf),
        ];
        for (op, reference) in unary {
            for &x in &periodic_pts {
                let mut a = ExprArena::new();
                let xv = a.push_var(0);
                let e = a.push_unary(op, xv);
                let pt = [x, 0.0, 0.0, 0.0];
                let got = eval(&a, e, &pt);
                let want = reference(x);
                assert_close_rel(got, want, &pt, TRIG_TOL);
            }
        }

        // The exponentials are checked purely relatively, on their own points.
        // `assert_close_rel`'s `want.abs().max(1.0)` floor turns into a plain
        // absolute 0.03 once the reference falls below 1, which is most of the
        // negative half-line here: `expf(-100)` is ~3.8e-44, so returning zero
        // would pass. And at +100 `expf` is `inf`, making `|got - inf| <= inf`
        // accept anything finite. A relative-only comparison over a range
        // where both sides stay finite and nonzero keeps every point binding.
        let exp_pts = [-20.0f32, -7.0, -0.6, 0.1, 0.6, 2.5, 7.0, 20.0];
        for (op, reference) in [
            (OpKind::Exp, libm::expf as Reference),
            (OpKind::Exp2, libm::exp2f as Reference),
        ] {
            for &x in &exp_pts {
                let mut a = ExprArena::new();
                let xv = a.push_var(0);
                let e = a.push_unary(op, xv);
                let pt = [x, 0.0, 0.0, 0.0];
                let got = eval(&a, e, &pt);
                let want = reference(x);
                assert!(
                    want.is_finite() && want > 0.0,
                    "{op:?} oracle at {x} is {want}: a relative check needs a finite, nonzero reference"
                );
                let rel_err = (got - want).abs() / want;
                assert!(
                    rel_err <= LIBM_TOL,
                    "at {pt:?}: {op:?} got {got}, want {want} (relative error {rel_err} > {LIBM_TOL})"
                );
            }
        }

        // Atan is unbounded; Asin/Acos are domain-restricted to [-1, 1].
        let atan_pts = [-100.0f32, -1.7, -0.3, 0.3, 1.7, 100.0];
        for &x in &atan_pts {
            let mut a = ExprArena::new();
            let xv = a.push_var(0);
            let e = a.push_unary(OpKind::Atan, xv);
            let pt = [x, 0.0, 0.0, 0.0];
            assert_close_rel(eval(&a, e, &pt), libm::atanf(x), &pt, LIBM_TOL);
        }
        let inverse_trig_pts = [-0.9f32, -0.5, -0.1, 0.1, 0.5, 0.9];
        for &x in &inverse_trig_pts {
            let mut a = ExprArena::new();
            let xv = a.push_var(0);
            let asin_e = a.push_unary(OpKind::Asin, xv);
            let pt = [x, 0.0, 0.0, 0.0];
            assert_close_rel(eval(&a, asin_e, &pt), libm::asinf(x), &pt, LIBM_TOL);

            let mut a = ExprArena::new();
            let xv = a.push_var(0);
            let acos_e = a.push_unary(OpKind::Acos, xv);
            assert_close_rel(eval(&a, acos_e, &pt), libm::acosf(x), &pt, LIBM_TOL);
        }

        // Ln/Log2/Log10's Cephes-style minimax fit is far tighter than the
        // trig/exp expansions above (worst case a few times 1e-7 relative,
        // vs. the percent-level `LIBM_TOL` those need), so hold it to its
        // own much narrower tolerance — loose enough for f32 rounding, tight
        // enough that a wrong Horner coefficient cannot hide inside it.
        const LOG_TOL: f32 = 3e-5;
        // The mantissa extraction reduces every input to the SAME fixed
        // range regardless of magnitude (`t ∈ [-0.293, 0.414]`, see
        // `expand_log2`), so a few magnitudes spanning decades sample almost
        // the same handful of `t` values — not enough to reliably land near
        // a Horner coefficient's worst point. Sweep the mantissa densely
        // (plus a couple of magnitudes to touch the exponent path, and the
        // range-reduction threshold itself at √2, each coefficient's most
        // sensitive point) instead.
        let mut log_pts: Vec<f32> = (0..256).map(|k| 1.0 + k as f32 * (0.999 / 256.0)).collect();
        log_pts.extend([1e-3f32, 10.0, 1e6, core::f32::consts::SQRT_2]);
        let logs: [(OpKind, Reference); 3] = [
            (OpKind::Ln, libm::logf),
            (OpKind::Log2, libm::log2f),
            (OpKind::Log10, libm::log10f),
        ];
        for (op, reference) in logs {
            for &x in &log_pts {
                let mut a = ExprArena::new();
                let xv = a.push_var(0);
                let e = a.push_unary(op, xv);
                let pt = [x, 0.0, 0.0, 0.0];
                let got = eval(&a, e, &pt);
                let want = reference(x);
                assert_close_rel(got, want, &pt, LOG_TOL);
            }
        }

        // Atan2 over all four quadrants plus the axis-aligned cases.
        for (y, x) in [
            (3.0f32, 4.0),
            (3.0, -4.0),
            (-3.0, 4.0),
            (-3.0, -4.0),
            (0.0, -1.0), // pi
            (1.0, 0.0),  // pi/2
        ] {
            let mut a = ExprArena::new();
            let yv = a.push_var(0);
            let xv = a.push_var(1);
            let e = a.push_binary(OpKind::Atan2, yv, xv);
            let pt = [y, x, 0.0, 0.0];
            let got = eval(&a, e, &pt);
            let want = libm::atan2f(y, x);
            assert_close_rel(got, want, &pt, LIBM_TOL);
        }

        // Pow needs a positive base (it lowers through log2/exp2).
        for (base, exp) in [(2.0f32, 3.3), (0.5, 2.0), (10.0, -1.5)] {
            let mut a = ExprArena::new();
            let bv = a.push_var(0);
            let ev = a.push_var(1);
            let e = a.push_binary(OpKind::Pow, bv, ev);
            let pt = [base, exp, 0.0, 0.0];
            let got = eval(&a, e, &pt);
            let want = libm::powf(base, exp);
            assert_close_rel(got, want, &pt, LIBM_TOL);
        }
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

        // A bare Gather (bound-memory read) cannot be differentiated, and
        // neither can its lowered RawGather form.
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

    #[test]
    fn differentiate_floor_ceil_and_round_to_zero_without_touching_their_operand() {
        // Floor/Ceil/Round are step functions: zero derivative, and — unlike
        // every other unary rule — the rule never reads the operand's own
        // derivative. Wrapping a Gather (itself undifferentiable) proves
        // that: if `push_deriv_children` wrongly marked the operand as
        // needing a derivative, the Gather's error would surface and this
        // would fail to lower at all instead of yielding 0.
        use crate::arena::{BufferDecl, BufferIdentity};
        for op in [OpKind::Floor, OpKind::Ceil, OpKind::Round] {
            let mut a = ExprArena::new();
            let b = a.declare_buffer(BufferDecl {
                id: BufferIdentity::mint(),
                width: 2,
                height: 1,
            });
            let gx = a.push_var(0);
            let zero = a.push_const(0.0);
            let g = a.push_gather(b, gx, zero);
            let e = a.push_unary(op, g);
            let (out, root) = lowered_derivative(&a, e, 0);
            let p = [0.0f32, 0.0, 0.0, 0.0];
            assert_close(eval(&out, root, &p), 0.0, &p);
        }
    }

    #[test]
    fn rebuild_copies_nary_children_slice_correctly() {
        // `copy_node`'s Nary arm reads `nodes_raw()[start..start+len]` — a
        // second Nary node makes `start` nonzero, which is what distinguishes
        // `start+len` from `start*len` (they coincide when start is 0).
        let mut a = ExprArena::new();
        let p = a.push_var(0);
        let _throwaway = a.push_nary(OpKind::Tuple, &[p]); // start=0, len=1

        let x = a.push_var(0);
        let y = a.push_var(1);
        let z = a.push_var(2);
        let root = a.push_nary(OpKind::Tuple, &[x, y, z]); // start=1, len=3

        // Any rebuild pass runs every reachable node through `copy_node` for
        // its non-matching arms; `expand_transcendentals` is the simplest
        // public one and this arena has nothing for it to actually lower.
        let new_root = expand_transcendentals(&mut a, root);
        let ExprNode::Nary(OpKind::Tuple, start, len) = a.node(new_root) else {
            panic!("expected a rebuilt Tuple, got {:?}", a.node(new_root));
        };
        let children = a.nary_children_slice(*start, *len);
        assert_eq!(children.len(), 3, "wrong slice length");
        for (child, expected_var) in children.iter().zip([0u8, 1, 2]) {
            assert!(
                matches!(a.node(*child), ExprNode::Var(v) if *v == expected_var),
                "child {child:?} should be Var({expected_var})"
            );
        }
    }

    #[test]
    fn legalize_lowers_gather() {
        use crate::arena::{BufferDecl, BufferIdentity};

        let mut a = ExprArena::new();
        let buf = a.declare_buffer(BufferDecl {
            id: BufferIdentity::mint(),
            width: 4,
            height: 1,
        });
        let x = a.push_var(0);
        let zero = a.push_const(0.0);
        let root = a.push_gather(buf, x, zero);

        let (out, out_root) = legalize(&a, root).expect("legalize");

        let mut seen = alloc::vec![false; out.nodes_raw().len()];
        let mut stack = alloc::vec![out_root];
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            assert!(
                !matches!(out.node(id), ExprNode::Ternary(OpKind::Gather, _, _, _)),
                "legalize left a high-level Gather reachable"
            );
            stack.extend(out.children(id));
        }

        let buf_data = [10.0f32, 20.0, 30.0, 40.0];
        let bindings = BindingTable::bind(&out, &[&buf_data[..]]).unwrap();
        assert_eq!(
            eval_scalar(&out, out_root, &[2.0, 0.0, 0.0, 0.0], &bindings),
            30.0
        );
    }

    #[test]
    fn legalize_lowers_reduce_transcendentals_and_dwrt_together() {
        // (Σ_{i<3} i) + d/dX[sin(X)], exercising `expand_reduce`, `lower_dwrt`,
        // and the `expand_transcendentals` pass that `lower_dwrt`'s own output
        // (a `Cos`) feeds into — all three passes `legalize` composes, on one
        // arena. `Dwrt` can only wrap what it can differentiate (`lower_dwrt`
        // has no rule for a raw `Reduce`, see `unsupported_op_errors_loudly`),
        // so the reduction and the derivative are independent subtrees joined
        // by `Add` rather than one nested inside the other.
        let mut a = ExprArena::new();
        let i = a.push_var(4);
        let red = a.push_reduce(OpKind::Add, 4, 3, i); // Σ_{i<3} i = 0+1+2 = 3

        let x = a.push_var(0);
        let s = a.push_unary(OpKind::Sin, x);
        let v0 = a.push_const(0.0);
        let dwrt_sin = a.push_binary(OpKind::Dwrt, s, v0); // d/dX sin(X) = cos(X)

        let root = a.push_binary(OpKind::Add, red, dwrt_sin);

        let (out, out_root) = legalize(&a, root).expect("legalize");

        let mut seen = alloc::vec![false; out.nodes_raw().len()];
        let mut stack = alloc::vec![out_root];
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            let node = out.node(id);
            assert!(
                !matches!(
                    node,
                    ExprNode::Nary(OpKind::Reduce, _, _)
                        | ExprNode::Unary(OpKind::Dwrt, _)
                        | ExprNode::Binary(OpKind::Dwrt, _, _)
                        | ExprNode::Ternary(OpKind::Dwrt, _, _, _)
                ),
                "legalize left a {node:?} reachable"
            );
            // Every transcendental, not just the input `Sin`: `lower_dwrt`
            // replaces that `Sin` with a `Cos`, so naming one op would let a
            // `legalize` that skipped its final expansion pass slip through.
            // The value check below cannot catch it either — `eval_scalar`
            // runs `expand_transcendentals_owned` itself.
            let leftover = match node {
                ExprNode::Unary(op, _) => is_transcendental_unary(*op),
                ExprNode::Binary(op, _, _) => is_transcendental_binary(*op),
                _ => false,
            };
            assert!(!leftover, "legalize left a backend-illegal {node:?}");
            stack.extend(out.children(id));
        }

        // 3 + cos(X), at X = 0.5.
        let want = 3.0 + 0.5f32.cos();
        let bindings = BindingTable::empty();
        let pt = [0.5f32, 0.0, 0.0, 0.0];
        let got = eval_scalar(&out, out_root, &pt, &bindings);
        assert_close_rel(got, want, &pt, 3e-2);
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
        if !arena.nodes_raw().iter().any(|n| {
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
        if !arena
            .nodes_raw()
            .iter()
            .any(|n| matches!(n, ExprNode::Nary(OpKind::Reduce, _, _)))
        {
            return Rewritten::Unchanged;
        }
        let mut owned = arena.clone();
        let new_root = expand_reduce(&mut owned, root);
        Rewritten::Changed(owned, new_root)
    }
}
