//! Lowering transcendental ops to primitive arithmetic subgraphs.
//!
//! Functions like `sin`, `cos`, `atan` have no single hardware instruction on
//! any target — they are *always* a polynomial of `mul`/`add`/`floor`/… So they
//! do not belong in a backend. Instead, before codegen, this pass rewrites each
//! transcendental [`ExprNode`] into the arena subgraph that computes it. The
//! result:
//!
//! - **No backend emits transcendental assembly.** x86/aarch64/AVX-512 (and any
//!   future backend) only ever see the primitive ops they already support.
//! - **Derivatives are free.** The jet (forward-mode AD) lowering differentiates
//!   arena arithmetic via the chain rule; an expanded `sin` differentiates with
//!   zero per-transcendental rules.
//! - **One source of truth.** The polynomial lives here, shared by every
//!   backend, with precision a property of this code (the polynomial degree),
//!   uniform across targets and tunable in one place.
//!
//! The pass runs *after* the e-graph optimizer (which may still reason about
//! `sin`/`cos` algebraically) and *before* any arena walk in the emitters, so
//! the transcendental `OpKind`s remain a valid authoring/optimization
//! vocabulary while never reaching machine code.
//!
//! Expansions are built from the jet-differentiable primitive set
//! (`Add`/`Sub`/`Mul`/`Neg`/`Sqrt`/`Floor`) — *not* `MulAdd` or `Select` — so
//! the derivative path keeps working. Fusing `mul`+`add` back into `MulAdd` is
//! the optimizer's job on the non-jet paths.

use crate::arena::{ExprArena, ExprId, ExprNode};
use crate::kind::OpKind;
use alloc::vec::Vec;

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
    matches!(op, OpKind::Atan2)
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

/// Convenience wrapper for the public `compile_arena_dag*` entries, which hold a
/// shared `&ExprArena`: clone it, expand transcendentals in the clone, and
/// return the owned arena + new root. Cheap when there are no transcendentals
/// (the clone is two `Vec`s and the walk just copies), so every entry can call
/// it unconditionally and be sure no backend — per-batch, scanline, or jet —
/// ever sees a transcendental node.
#[must_use]
pub fn expand_transcendentals_owned(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
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
pub fn expand_gather_owned(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
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

    // xi = clamp(floor(x), 0, width-1); yi = clamp(floor(y), 0, height-1)
    let fx = arena.push_unary(OpKind::Floor, x);
    let xi = arena.push_ternary(OpKind::Clamp, fx, zero, max_x);
    let fy = arena.push_unary(OpKind::Floor, y);
    let yi = arena.push_ternary(OpKind::Clamp, fy, zero, max_y);

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

    // acc = body[var:=0]; then acc = combiner(acc, body[var:=k]) for k in 1..N.
    let mut acc = substitute_var(arena, body, var_idx, 0.0);
    for k in 1..n {
        let term = substitute_var(arena, body, var_idx, k as f32);
        acc = arena.push_binary(combiner_op, acc, term);
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

/// Clone the subtree at `root`, replacing every `Var(var)` with `Const(value)`.
/// Shared nodes are rebuilt once (memoized), so a DAG body stays a DAG.
fn substitute_var(arena: &mut ExprArena, root: ExprId, var: u8, value: f32) -> ExprId {
    let n = arena.nodes_raw().len();
    let mut memo: Vec<Option<ExprId>> = alloc::vec![None; n];
    subst_rec(arena, root, var, value, &mut memo)
}

fn subst_rec(
    arena: &mut ExprArena,
    id: ExprId,
    var: u8,
    value: f32,
    memo: &mut Vec<Option<ExprId>>,
) -> ExprId {
    let idx = id.0 as usize;
    if let Some(Some(m)) = memo.get(idx) {
        return *m;
    }
    let new = match arena.node(id).clone() {
        ExprNode::Var(i) if i == var => arena.push_const(value),
        ExprNode::Var(i) => arena.push_var(i),
        ExprNode::Const(v) => arena.push_const(v),
        ExprNode::Param(i) => arena.push_param(i),
        ExprNode::Buffer(b) => arena.push_buffer(b),
        ExprNode::Unary(op, a) => {
            let a = subst_rec(arena, a, var, value, memo);
            arena.push_unary(op, a)
        }
        ExprNode::Binary(op, a, b) => {
            let a = subst_rec(arena, a, var, value, memo);
            let b = subst_rec(arena, b, var, value, memo);
            arena.push_binary(op, a, b)
        }
        ExprNode::Ternary(op, a, b, c) => {
            let a = subst_rec(arena, a, var, value, memo);
            let b = subst_rec(arena, b, var, value, memo);
            let c = subst_rec(arena, c, var, value, memo);
            arena.push_ternary(op, a, b, c)
        }
        ExprNode::Nary(op, start, len) => {
            let (s, l) = (start as usize, len as usize);
            let children: Vec<ExprId> = arena.nary_children_raw()[s..s + l].to_vec();
            let mapped: Vec<ExprId> = children
                .into_iter()
                .map(|ch| subst_rec(arena, ch, var, value, memo))
                .collect();
            arena.push_nary(op, &mapped)
        }
    };
    if idx < memo.len() {
        memo[idx] = Some(new);
    }
    new
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
fn differentiate(arena: &mut ExprArena, expr: ExprId, var: u8) -> Result<ExprId, &'static str> {
    let mut memo: Vec<Option<ExprId>> = alloc::vec![None; arena.nodes_raw().len()];
    diff_rec(arena, expr, var, &mut memo)
}

fn diff_rec(
    arena: &mut ExprArena,
    id: ExprId,
    var: u8,
    memo: &mut Vec<Option<ExprId>>,
) -> Result<ExprId, &'static str> {
    let idx = id.0 as usize;
    if let Some(Some(d)) = memo.get(idx) {
        return Ok(*d);
    }
    let d = diff_node(arena, id, var, memo)?;
    if idx < memo.len() {
        memo[idx] = Some(d);
    }
    Ok(d)
}

fn diff_node(
    arena: &mut ExprArena,
    id: ExprId,
    var: u8,
    memo: &mut Vec<Option<ExprId>>,
) -> Result<ExprId, &'static str> {
    match arena.node(id).clone() {
        ExprNode::Var(i) => Ok(arena.push_const(if i == var { 1.0 } else { 0.0 })),
        // Constants and scalar params (baked before evaluation) are
        // coordinate-independent.
        ExprNode::Const(_) | ExprNode::Param(_) => Ok(arena.push_const(0.0)),
        ExprNode::Buffer(_) => Err("lower_dwrt: cannot differentiate a bound-memory read"),

        ExprNode::Unary(op, a) => {
            let du = diff_rec(arena, a, var, memo)?;
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
                // Step functions: zero derivative almost everywhere.
                OpKind::Floor | OpKind::Ceil | OpKind::Round => Ok(arena.push_const(0.0)),
                // fract(u) = u − floor(u), so d = u' a.e.
                OpKind::Fract => Ok(du),
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
                OpKind::TruncToInt | OpKind::IntToFloat => {
                    Err("lower_dwrt: cannot differentiate integer/bit-manipulation ops")
                }
                _ => Err("lower_dwrt: no derivative rule for this unary op"),
            }
        }

        ExprNode::Binary(op, a, b) => match op {
            OpKind::Add => {
                let da = diff_rec(arena, a, var, memo)?;
                let db = diff_rec(arena, b, var, memo)?;
                Ok(d_add(arena, da, db))
            }
            OpKind::Sub => {
                let da = diff_rec(arena, a, var, memo)?;
                let db = diff_rec(arena, b, var, memo)?;
                Ok(d_sub(arena, da, db))
            }
            // Product rule.
            OpKind::Mul => {
                let da = diff_rec(arena, a, var, memo)?;
                let db = diff_rec(arena, b, var, memo)?;
                let t1 = d_mul(arena, da, b);
                let t2 = d_mul(arena, a, db);
                Ok(d_add(arena, t1, t2))
            }
            // Quotient rule: (a'b − ab') / b².
            OpKind::Div => {
                let da = diff_rec(arena, a, var, memo)?;
                let db = diff_rec(arena, b, var, memo)?;
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
                let da = diff_rec(arena, a, var, memo)?;
                let db = diff_rec(arena, b, var, memo)?;
                let mask = arena.push_binary(OpKind::Lt, a, b);
                Ok(arena.push_ternary(OpKind::Select, mask, da, db))
            }
            OpKind::Max => {
                let da = diff_rec(arena, a, var, memo)?;
                let db = diff_rec(arena, b, var, memo)?;
                let mask = arena.push_binary(OpKind::Gt, a, b);
                Ok(arena.push_ternary(OpKind::Select, mask, da, db))
            }
            // Masks are step functions: zero derivative almost everywhere.
            OpKind::Lt | OpKind::Le | OpKind::Gt | OpKind::Ge | OpKind::Eq | OpKind::Ne => {
                Ok(arena.push_const(0.0))
            }
            // d(atan2(y, x)) = (x·y' − y·x') / (x² + y²).
            OpKind::Atan2 => {
                let dy = diff_rec(arena, a, var, memo)?;
                let dx = diff_rec(arena, b, var, memo)?;
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
            // d(hypot(a, b)) = (a·a' + b·b') / hypot(a, b).
            OpKind::Hypot => {
                let da = diff_rec(arena, a, var, memo)?;
                let db = diff_rec(arena, b, var, memo)?;
                let t1 = d_mul(arena, a, da);
                let t2 = d_mul(arena, b, db);
                let num = d_add(arena, t1, t2);
                if is_const_zero(arena, num) {
                    return Ok(num);
                }
                let h = arena.push_binary(OpKind::Hypot, a, b);
                Ok(arena.push_binary(OpKind::Div, num, h))
            }
            // d(f^g) = f^g · (g'·ln f + g·f'/f)  (Jet2's rule).
            OpKind::Pow => {
                let df = diff_rec(arena, a, var, memo)?;
                let dg = diff_rec(arena, b, var, memo)?;
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
                let da = diff_rec(arena, a, var, memo)?;
                let db = diff_rec(arena, b, var, memo)?;
                let dc = diff_rec(arena, c, var, memo)?;
                let t1 = d_mul(arena, da, b);
                let t2 = d_mul(arena, a, db);
                let prod = d_add(arena, t1, t2);
                Ok(d_add(arena, prod, dc))
            }
            // Blend the branch derivatives on the primal mask (Jet2 select).
            OpKind::Select => {
                let db = diff_rec(arena, b, var, memo)?;
                let dc = diff_rec(arena, c, var, memo)?;
                Ok(arena.push_ternary(OpKind::Select, a, db, dc))
            }
            // clamp(x, lo, hi) = min(max(x, lo), hi); differentiate that exact
            // composition so masks (and tie behavior) match the Jet2 chain.
            OpKind::Clamp => {
                let dx = diff_rec(arena, a, var, memo)?;
                let dlo = diff_rec(arena, b, var, memo)?;
                let dhi = diff_rec(arena, c, var, memo)?;
                let gt = arena.push_binary(OpKind::Gt, a, b);
                let dm = arena.push_ternary(OpKind::Select, gt, dx, dlo);
                let m = arena.push_binary(OpKind::Max, a, b);
                let lt = arena.push_binary(OpKind::Lt, m, c);
                Ok(arena.push_ternary(OpKind::Select, lt, dm, dhi))
            }
            OpKind::Gather => Err("lower_dwrt: cannot differentiate a bound-memory read"),
            _ => Err("lower_dwrt: no derivative rule for this ternary op"),
        },

        ExprNode::Nary(_, _, _) => {
            Err("lower_dwrt: cannot differentiate an Nary op (Reduce/Tuple)")
        }
    }
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
        // cos(x) = sin(x + π/2)
        OpKind::Cos => {
            let half_pi = arena.push_const(core::f32::consts::FRAC_PI_2);
            let shifted = arena.push_binary(OpKind::Add, arg, half_pi);
            expand_sin(arena, shifted)
        }
        // tan(x) = sin(x) / cos(x). Expand both so neither reaches a backend.
        OpKind::Tan => {
            let s = expand_sin(arena, arg);
            let half_pi = arena.push_const(core::f32::consts::FRAC_PI_2);
            let shifted = arena.push_binary(OpKind::Add, arg, half_pi);
            let c = expand_sin(arena, shifted);
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
        _ => unreachable!("expand_binary called on non-transcendental {op:?}"),
    }
}

/// `atan2(y, x)` (four-quadrant) as a primitive subgraph.
///
/// Mirrors the runtime Compounds version: reduce to a ratio in [-1,1] (swapping
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
    let c1 = arena.push_const(1.0);
    let c3 = arena.push_const(-0.333_333_33);
    let c5 = arena.push_const(0.2);
    let c7 = arena.push_const(-0.142_857_14);
    let p = horner_step(arena, c7, r2, c5);
    let p = horner_step(arena, p, r2, c3);
    let p = horner_step(arena, p, r2, c1);
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

/// `sin(x)` as a primitive subgraph (Chebyshev, matching the runtime path).
///
/// Range-reduce to `[-π, π]`, normalize to `[-1, 1]`, then a degree-7 odd
/// Chebyshev polynomial. Built from `Add`/`Sub`/`Mul`/`Floor` only (no `MulAdd`/
/// `Select`), so the jet path differentiates it via the chain rule.
fn expand_sin(arena: &mut ExprArena, x: ExprId) -> ExprId {
    use core::f32::consts::{PI, TAU};

    // k = floor(x / 2π + 0.5)
    let two_pi_inv = arena.push_const(1.0 / TAU);
    let half = arena.push_const(0.5);
    let xr = arena.push_binary(OpKind::Mul, x, two_pi_inv);
    let xr = arena.push_binary(OpKind::Add, xr, half);
    let k = arena.push_unary(OpKind::Floor, xr);

    // xx = x - k·2π
    let tau = arena.push_const(TAU);
    let k_tau = arena.push_binary(OpKind::Mul, k, tau);
    let xx = arena.push_binary(OpKind::Sub, x, k_tau);

    // t = xx / π
    let pi_inv = arena.push_const(1.0 / PI);
    let t = arena.push_binary(OpKind::Mul, xx, pi_inv);

    // t2 = t·t
    let t2 = arena.push_binary(OpKind::Mul, t, t);

    // Horner: p = ((c7·t2 + c5)·t2 + c3)·t2 + c1, expanded as mul+add.
    let c1 = arena.push_const(core::f32::consts::PI);
    let c3 = arena.push_const(-5.167_712_7);
    let c5 = arena.push_const(2.550_164);
    let c7 = arena.push_const(-0.599_264_5);
    let p = horner_step(arena, c7, t2, c5);
    let p = horner_step(arena, p, t2, c3);
    let p = horner_step(arena, p, t2, c1);

    // sin ≈ t·p
    arena.push_binary(OpKind::Mul, t, p)
}

/// `acc·x + add` as `Add(Mul(acc, x), add)` — plain mul+add so the jet path
/// (which has no `MulAdd` rule) can differentiate it. The optimizer re-fuses to
/// `MulAdd` on the non-jet paths.
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
        let tol = 1e-3 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "at {pt:?}: got {got}, want {want} (tol {tol})"
        );
    }

    #[test]
    fn d_var_is_one_or_zero() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let (out, root) = lowered_derivative(&a, x, 0);
        assert_close(eval(&out, root, &[3.0, 5.0, 0.0, 0.0]), 1.0, &[3.0, 5.0, 0.0, 0.0]);

        let mut a = ExprArena::new();
        let y = a.push_var(1);
        let (out, root) = lowered_derivative(&a, y, 0);
        assert_close(eval(&out, root, &[3.0, 5.0, 0.0, 0.0]), 0.0, &[3.0, 5.0, 0.0, 0.0]);
    }

    #[test]
    fn d_sqrt_sum_of_squares() {
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
    fn d_min_max_pick_branch_derivative() {
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
        assert_close(eval(&out, root, &[1.0, 5.0, 0.0, 0.0]), 2.0, &[1.0, 5.0, 0.0, 0.0]);
        assert_close(eval(&out, root, &[9.0, 1.0, 0.0, 0.0]), 0.0, &[9.0, 1.0, 0.0, 0.0]);

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let two = a.push_const(2.0);
        let three = a.push_const(3.0);
        let x2 = a.push_binary(OpKind::Mul, x, two);
        let y3 = a.push_binary(OpKind::Mul, y, three);
        let e = a.push_binary(OpKind::Max, x2, y3);
        let (out, root) = lowered_derivative(&a, e, 0);
        assert_close(eval(&out, root, &[9.0, 1.0, 0.0, 0.0]), 2.0, &[9.0, 1.0, 0.0, 0.0]);
        assert_close(eval(&out, root, &[1.0, 5.0, 0.0, 0.0]), 0.0, &[1.0, 5.0, 0.0, 0.0]);
    }

    #[test]
    fn d_select_blends_branch_derivatives() {
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
        assert_close(eval(&out, root, &[3.0, 1.0, 0.0, 0.0]), 6.0, &[3.0, 1.0, 0.0, 0.0]);
        assert_close(eval(&out, root, &[3.0, -1.0, 0.0, 0.0]), 5.0, &[3.0, -1.0, 0.0, 0.0]);
    }

    #[test]
    fn d_clamp_saturates() {
        // d/dx clamp(x·x, 0, 10): 2x inside, 0 once saturated.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let zero = a.push_const(0.0);
        let ten = a.push_const(10.0);
        let xx = a.push_binary(OpKind::Mul, x, x);
        let e = a.push_ternary(OpKind::Clamp, xx, zero, ten);
        let (out, root) = lowered_derivative(&a, e, 0);
        assert_close(eval(&out, root, &[2.0, 0.0, 0.0, 0.0]), 4.0, &[2.0, 0.0, 0.0, 0.0]);
        assert_close(eval(&out, root, &[5.0, 0.0, 0.0, 0.0]), 0.0, &[5.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn d_mul_add_matches_product_rule() {
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
    fn d_transcendentals() {
        // d/dx sin(x) = cos(x); d/dx exp(x·x) = 2x·exp(x²); d/dx ln(x) = 1/x.
        let pts = [[0.7f32, 0.0, 0.0, 0.0], [1.3, 0.0, 0.0, 0.0]];

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let e = a.push_unary(OpKind::Sin, x);
        let (out, root) = lowered_derivative(&a, e, 0);
        for p in &pts {
            assert_close(eval(&out, root, p), p[0].cos(), p);
        }

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let xx = a.push_binary(OpKind::Mul, x, x);
        let e = a.push_unary(OpKind::Exp, xx);
        let (out, root) = lowered_derivative(&a, e, 0);
        for p in &pts {
            assert_close(eval(&out, root, p), 2.0 * p[0] * (p[0] * p[0]).exp(), p);
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

    #[test]
    fn unsupported_op_errors_loudly() {
        // Differentiating a Reduce has no rule: the pass must refuse.
        let mut a = ExprArena::new();
        let combiner = a.push_const(OpKind::Add as u8 as f32);
        let rvar = a.push_const(0.0);
        let extent = a.push_const(4.0);
        let body = a.push_var(4);
        let red = a.push_nary(OpKind::Reduce, &[combiner, rvar, extent, body]);
        let v0 = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, red, v0);
        assert!(lower_dwrt_owned(&a, root).is_err());
    }
}
