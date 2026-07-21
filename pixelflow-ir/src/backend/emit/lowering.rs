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
use core::fmt;

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
    let result: Result<ExprId, core::convert::Infallible> =
        try_rebuild_arena(arena, root, |arena, node, m| Ok(lower(arena, node, m)));
    match result {
        Ok(id) => id,
        Err(never) => match never {},
    }
}

/// Fallible variant of [`rebuild_arena`]: the `lower` hook may abort the whole
/// rebuild with an error (used by [`lower_dwrt`], whose per-op derivative rules
/// are partial). Same traversal, same dedup, same contract otherwise.
fn try_rebuild_arena<F, E>(arena: &mut ExprArena, root: ExprId, mut lower: F) -> Result<ExprId, E>
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
/// `log2(x) = e + log2(m)` where `e` is the unbiased exponent and `m ∈ [1,2)` is
/// the mantissa. Extract `e` by shifting the exponent field down; rebuild `m` by
/// masking the mantissa bits and OR-ing in exponent bias 127 (= 1.0). Then a
/// degree-4 polynomial for `log2(m)` on `[1,2)`.
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

    // log2(m) on [1,2): polynomial in (m - 1).
    let one = arena.push_const(1.0);
    let t = arena.push_binary(OpKind::Sub, m, one);
    let c1 = arena.push_const(core::f32::consts::LOG2_E);
    let c2 = arena.push_const(-0.721_347_5);
    let c3 = arena.push_const(0.479_924_46);
    let c4 = arena.push_const(-0.298_768_3);
    let p = horner_step(arena, c4, t, c3);
    let p = horner_step(arena, p, t, c2);
    let p = horner_step(arena, p, t, c1);
    let log2_m = arena.push_binary(OpKind::Mul, p, t);

    arena.push_binary(OpKind::Add, e, log2_m)
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

// ────────────────────────────── Dwrt lowering ─────────────────────────────────

/// Failure of a lowering pass whose per-op rules are partial (today: only
/// [`lower_dwrt`]). Every variant is a loud, named failure — no rule ever
/// silently produces a wrong-but-plausible expression.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LowerError {
    /// `Dwrt` was applied over an op with no symbolic differentiation rule.
    UnsupportedOp {
        /// The op that has no derivative rule.
        op: OpKind,
    },
    /// The `var` operand of a `Dwrt` node was not a `Const` coordinate index.
    DwrtVarNotConst,
    /// The `var` operand of a `Dwrt` was a `Const` outside `0..4` (X/Y/Z/W).
    DwrtVarOutOfRange {
        /// The out-of-range value found in the `Const` operand.
        value: f32,
    },
    /// Differentiation recursed deeper than [`MAX_DERIV_DEPTH`] — either a
    /// pathologically deep expression or runaway nesting. Erroring here is the
    /// loud alternative to a stack overflow.
    DepthExceeded {
        /// The bound that was exceeded ([`MAX_DERIV_DEPTH`]).
        limit: usize,
    },
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedOp { op } => write!(
                f,
                "lower_dwrt: no symbolic derivative rule for op '{}'",
                op.name()
            ),
            Self::DwrtVarNotConst => {
                write!(f, "lower_dwrt: Dwrt var operand must be a Const index")
            }
            Self::DwrtVarOutOfRange { value } => write!(
                f,
                "lower_dwrt: Dwrt var index {value} out of range (must be 0..4 = X/Y/Z/W)"
            ),
            Self::DepthExceeded { limit } => write!(
                f,
                "lower_dwrt: differentiation exceeded the depth bound of {limit} nodes"
            ),
        }
    }
}

impl core::error::Error for LowerError {}

impl LowerError {
    /// Static rendering for the `Result<_, &'static str>` compile entries.
    ///
    /// The compile drivers return `&'static str` errors, which cannot carry a
    /// formatted op name; this maps each error to the most specific static
    /// message available. [`fmt::Display`] remains the precise form.
    #[must_use]
    pub fn as_static_str(self) -> &'static str {
        match self {
            Self::UnsupportedOp { op } => match op {
                OpKind::TruncToInt => "lower_dwrt: no derivative rule for 'trunc_to_int'",
                OpKind::IntToFloat => "lower_dwrt: no derivative rule for 'int_to_float'",
                OpKind::IAdd => "lower_dwrt: no derivative rule for 'iadd'",
                OpKind::Shl => "lower_dwrt: no derivative rule for 'shl'",
                OpKind::Shr => "lower_dwrt: no derivative rule for 'shr'",
                OpKind::BitAnd => "lower_dwrt: no derivative rule for 'bitand'",
                OpKind::BitOr => "lower_dwrt: no derivative rule for 'bitor'",
                OpKind::Tuple => "lower_dwrt: no derivative rule for 'tuple'",
                OpKind::Reduce => "lower_dwrt: no derivative rule for 'reduce'",
                OpKind::Buffer => "lower_dwrt: no derivative rule for a bare 'buffer' leaf",
                _ => "lower_dwrt: no derivative rule for this op",
            },
            Self::DwrtVarNotConst => "lower_dwrt: Dwrt var operand must be a Const index",
            Self::DwrtVarOutOfRange { .. } => "lower_dwrt: Dwrt var index out of range (0..4)",
            Self::DepthExceeded { .. } => "lower_dwrt: differentiation depth bound exceeded",
        }
    }
}

/// Recursion bound for [`lower_dwrt`]'s differentiation walk. Deeper
/// expressions produce [`LowerError::DepthExceeded`] instead of a stack
/// overflow. Real kernels are well under this; the scanline JIT rejects
/// anything past depth 64 anyway.
pub const MAX_DERIV_DEPTH: usize = 512;

/// Eliminate every `Dwrt` node reachable from `root` by symbolic
/// differentiation, returning the (possibly new) root in the same arena.
///
/// `Dwrt(expr, Const(i))` (the [`OpKind::Dwrt`] operator built by `D(expr,
/// var)` / the `DX`/`DY`/`DZ` accessors) becomes an ordinary arithmetic
/// expression in the same OpKinds — the derivative is just another kernel,
/// scheduled and emitted by the same backend. This is the sanctioned
/// replacement for the deleted emit-time jet mode (commit bd47aa7c): lowering
/// happens *before* scheduling, not inside a special evaluator.
///
/// Nodes are rewritten post-order (innermost first), so nested `Dwrt` — second
/// derivatives like `Dwrt(Dwrt(e, 0), 1)` — see an already-derivative-free
/// operand and need no fixpoint iteration. Value subexpressions are referenced
/// by their existing ids (never copied), and each `(node, var)` pair is
/// differentiated once, so a DAG stays a DAG.
///
/// # Derivative conventions (matching `pixelflow_core::Jet2` semantics)
///
/// - `Gather`/`RawGather` differentiate to **0**: a buffer lookup is piecewise
///   constant in its index (a hard step at every texel boundary), so 0 is the
///   almost-everywhere derivative, exactly as `Floor` (which the index math is
///   built from) differentiates to 0.
/// - Comparisons (`Lt`..`Ne`) and `Floor`/`Ceil`/`Round` are step functions:
///   derivative 0 almost everywhere.
/// - `Abs`, `Min`/`Max`, `Clamp`, `Select` are piecewise: the derivative
///   selects the active branch's derivative (`Select` on the same condition
///   Jet2 uses: `Lt` for min, `Gt` for max).
/// - `Param` is a bound scalar constant with respect to coordinates:
///   derivative 0.
///
/// Any op with no rule — the integer/bit-manipulation ops, `Reduce`, `Tuple` —
/// returns [`LowerError::UnsupportedOp`] naming the op. Never a silent 0.
pub fn lower_dwrt(arena: &mut ExprArena, root: ExprId) -> Result<ExprId, LowerError> {
    let new_root = try_rebuild_arena(arena, root, |arena, node, m| match node {
        ExprNode::Binary(OpKind::Dwrt, expr, var) => {
            let var = dwrt_var_index(arena, *var)?;
            // `m(*expr)` is the already-lowered operand: post-order guarantees
            // any inner `Dwrt` in it has been eliminated.
            let mut memo = alloc::collections::BTreeMap::new();
            let d = differentiate(arena, m(*expr), var, 0, &mut memo)?;
            Ok(Some(d))
        }
        _ => Ok(None),
    })?;

    // Post-order elimination is total, so a surviving Dwrt is an internal
    // logic error in this pass — check loudly rather than let it reach the
    // scheduler's (now unreachable-precondition) assert.
    debug_assert!(
        !subtree_contains_dwrt(arena, new_root),
        "lower_dwrt: internal error — a Dwrt survived lowering"
    );
    Ok(new_root)
}

/// Owned wrapper mirroring [`expand_transcendentals_owned`]: identity fast-path
/// when the arena has no `Dwrt`, otherwise clone-and-lower. Every public
/// compile entry calls this first, so no `Dwrt` ever reaches scheduling.
pub fn lower_dwrt_owned(
    arena: &ExprArena,
    root: ExprId,
) -> Result<(ExprArena, ExprId), LowerError> {
    if !arena
        .nodes_raw()
        .iter()
        .any(|n| matches!(n, ExprNode::Binary(OpKind::Dwrt, _, _)))
    {
        return Ok((arena.clone(), root));
    }
    let mut owned = arena.clone();
    let new_root = lower_dwrt(&mut owned, root)?;
    Ok((owned, new_root))
}

/// Whether any `Dwrt` node is reachable from `root`.
fn subtree_contains_dwrt(arena: &ExprArena, root: ExprId) -> bool {
    let mut stack = alloc::vec![root];
    let mut seen = alloc::vec![false; arena.len()];
    while let Some(id) = stack.pop() {
        if core::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        if matches!(arena.node(id), ExprNode::Binary(OpKind::Dwrt, _, _)) {
            return true;
        }
        for c in arena.children(id) {
            stack.push(c);
        }
    }
    false
}

/// Read and validate the differentiation-variable operand of a `Dwrt` node.
fn dwrt_var_index(arena: &ExprArena, var: ExprId) -> Result<u8, LowerError> {
    let ExprNode::Const(v) = *arena.node(var) else {
        return Err(LowerError::DwrtVarNotConst);
    };
    if v.fract() != 0.0 || !(0.0..4.0).contains(&v) {
        return Err(LowerError::DwrtVarOutOfRange { value: v });
    }
    Ok(v as u8)
}

/// Is this node literally `Const(0.0)`?
fn is_const_zero(arena: &ExprArena, id: ExprId) -> bool {
    matches!(arena.node(id), ExprNode::Const(v) if *v == 0.0)
}

/// `a + b` with additive-identity folding (keeps derivative DAGs compact —
/// most partials of most subtrees are statically zero).
fn dadd(arena: &mut ExprArena, a: ExprId, b: ExprId) -> ExprId {
    if is_const_zero(arena, a) {
        return b;
    }
    if is_const_zero(arena, b) {
        return a;
    }
    arena.push_binary(OpKind::Add, a, b)
}

/// `a - b` with identity folding.
fn dsub(arena: &mut ExprArena, a: ExprId, b: ExprId) -> ExprId {
    if is_const_zero(arena, b) {
        return a;
    }
    if is_const_zero(arena, a) {
        return arena.push_unary(OpKind::Neg, b);
    }
    arena.push_binary(OpKind::Sub, a, b)
}

/// `a * b` with annihilator folding (`0 * x = 0`).
///
/// This folds `0 * NaN`/`0 * inf` to `0`, i.e. the *strong* zero the selected
/// piecewise rules already assume — the same choice `Jet2`'s statically-zero
/// partials make.
fn dmul(arena: &mut ExprArena, a: ExprId, b: ExprId) -> ExprId {
    if is_const_zero(arena, a) || is_const_zero(arena, b) {
        return arena.push_const(0.0);
    }
    if matches!(arena.node(a), ExprNode::Const(v) if *v == 1.0) {
        return b;
    }
    if matches!(arena.node(b), ExprNode::Const(v) if *v == 1.0) {
        return a;
    }
    arena.push_binary(OpKind::Mul, a, b)
}

/// `-a` with `-0 = 0` folding.
fn dneg(arena: &mut ExprArena, a: ExprId) -> ExprId {
    if is_const_zero(arena, a) {
        return a;
    }
    arena.push_unary(OpKind::Neg, a)
}

/// `a / b` with a strong-zero numerator fold (see [`dmul`]).
fn ddiv(arena: &mut ExprArena, a: ExprId, b: ExprId) -> ExprId {
    if is_const_zero(arena, a) {
        return a;
    }
    arena.push_binary(OpKind::Div, a, b)
}

/// Symbolic derivative of the (already `Dwrt`-free) subtree at `id` with
/// respect to coordinate `var`, memoized per `(node, var)` so shared
/// subexpressions differentiate once. `depth` is the recursion depth, bounded
/// by [`MAX_DERIV_DEPTH`].
fn differentiate(
    arena: &mut ExprArena,
    id: ExprId,
    var: u8,
    depth: usize,
    memo: &mut alloc::collections::BTreeMap<(u32, u8), ExprId>,
) -> Result<ExprId, LowerError> {
    if depth > MAX_DERIV_DEPTH {
        return Err(LowerError::DepthExceeded {
            limit: MAX_DERIV_DEPTH,
        });
    }
    if let Some(&d) = memo.get(&(id.0, var)) {
        return Ok(d);
    }

    let node = arena.node(id).clone();
    let d = match node {
        // ── Leaves ──
        ExprNode::Var(i) => {
            let v = if i == var { 1.0 } else { 0.0 };
            arena.push_const(v)
        }
        ExprNode::Const(_) => arena.push_const(0.0),
        // A bound scalar parameter is constant with respect to coordinates.
        ExprNode::Param(_) => arena.push_const(0.0),
        // A bare Buffer is not a value (it only appears under Gather /
        // RawGather, both handled below without recursing into it).
        ExprNode::Buffer(_) => {
            return Err(LowerError::UnsupportedOp { op: OpKind::Buffer });
        }

        ExprNode::Unary(op, a) => {
            let du = differentiate(arena, a, var, depth + 1, memo)?;
            match op {
                OpKind::Neg => dneg(arena, du),
                // (√u)' = u' · ½·rsqrt(u)   (Jet2's rsqrt form)
                OpKind::Sqrt => {
                    let r = arena.push_unary(OpKind::Rsqrt, a);
                    let half = arena.push_const(0.5);
                    let f = dmul(arena, half, r);
                    dmul(arena, f, du)
                }
                // (u^-½)' = -½·u^-3/2 · u' = -½·rsqrt(u)³ · u'
                OpKind::Rsqrt => {
                    let r = arena.push_unary(OpKind::Rsqrt, a);
                    let r2 = dmul(arena, r, r);
                    let r3 = dmul(arena, r2, r);
                    let neg_half = arena.push_const(-0.5);
                    let f = dmul(arena, neg_half, r3);
                    dmul(arena, f, du)
                }
                // |u|' : piecewise on sign — Select(u >= 0, u', -u').
                OpKind::Abs => {
                    let zero = arena.push_const(0.0);
                    let nonneg = arena.push_binary(OpKind::Ge, a, zero);
                    let ndu = dneg(arena, du);
                    select_branch(arena, nonneg, du, ndu)
                }
                // (1/u)' = -u' / u²
                OpKind::Recip => {
                    let ndu = dneg(arena, du);
                    let u2 = dmul(arena, a, a);
                    ddiv(arena, ndu, u2)
                }
                // Step functions: derivative 0 almost everywhere.
                OpKind::Floor | OpKind::Ceil | OpKind::Round => arena.push_const(0.0),
                // fract(u) = u - floor(u): derivative u' almost everywhere.
                OpKind::Fract => du,
                OpKind::Sin => {
                    let c = arena.push_unary(OpKind::Cos, a);
                    dmul(arena, c, du)
                }
                OpKind::Cos => {
                    let s = arena.push_unary(OpKind::Sin, a);
                    let ns = dneg(arena, s);
                    dmul(arena, ns, du)
                }
                // (tan u)' = u' / cos²u
                OpKind::Tan => {
                    let c = arena.push_unary(OpKind::Cos, a);
                    let c2 = dmul(arena, c, c);
                    ddiv(arena, du, c2)
                }
                // (asin u)' = u' / √(1-u²)
                OpKind::Asin => {
                    let s = one_minus_sq_sqrt(arena, a);
                    ddiv(arena, du, s)
                }
                // (acos u)' = -u' / √(1-u²)
                OpKind::Acos => {
                    let s = one_minus_sq_sqrt(arena, a);
                    let q = ddiv(arena, du, s);
                    dneg(arena, q)
                }
                // (atan u)' = u' / (1+u²)
                OpKind::Atan => {
                    let one = arena.push_const(1.0);
                    let u2 = dmul(arena, a, a);
                    let den = dadd(arena, one, u2);
                    ddiv(arena, du, den)
                }
                OpKind::Exp => {
                    let e = arena.push_unary(OpKind::Exp, a);
                    dmul(arena, e, du)
                }
                // (2^u)' = 2^u · ln2 · u'
                OpKind::Exp2 => {
                    let e = arena.push_unary(OpKind::Exp2, a);
                    let ln2 = arena.push_const(core::f32::consts::LN_2);
                    let f = dmul(arena, e, ln2);
                    dmul(arena, f, du)
                }
                // (ln u)' = u' / u
                OpKind::Ln => ddiv(arena, du, a),
                // (log2 u)' = u' / (u·ln2)
                OpKind::Log2 => {
                    let ln2 = arena.push_const(core::f32::consts::LN_2);
                    let den = dmul(arena, a, ln2);
                    ddiv(arena, du, den)
                }
                // (log10 u)' = u' / (u·ln10)
                OpKind::Log10 => {
                    let ln10 = arena.push_const(core::f32::consts::LN_10);
                    let den = dmul(arena, a, ln10);
                    ddiv(arena, du, den)
                }
                // Integer/bit-domain ops have no derivative. Loud error.
                _ => return Err(LowerError::UnsupportedOp { op }),
            }
        }

        ExprNode::Binary(op, a, b) => match op {
            // Nested derivative that this walk itself uncovered (a `Dwrt`
            // stored behind a shared node the rebuild has not visited): take
            // the inner derivative first, then differentiate the result.
            OpKind::Dwrt => {
                let inner_var = dwrt_var_index(arena, b)?;
                let inner = differentiate(arena, a, inner_var, depth + 1, memo)?;
                differentiate(arena, inner, var, depth + 1, memo)?
            }
            // Buffer reads are piecewise constant in their indices (a hard
            // step at every texel): derivative 0, matching Floor. See module
            // docs on `lower_dwrt`.
            OpKind::RawGather => arena.push_const(0.0),
            OpKind::Add | OpKind::Sub => {
                let da = differentiate(arena, a, var, depth + 1, memo)?;
                let db = differentiate(arena, b, var, depth + 1, memo)?;
                if op == OpKind::Add {
                    dadd(arena, da, db)
                } else {
                    dsub(arena, da, db)
                }
            }
            // Product rule.
            OpKind::Mul => {
                let da = differentiate(arena, a, var, depth + 1, memo)?;
                let db = differentiate(arena, b, var, depth + 1, memo)?;
                let t1 = dmul(arena, da, b);
                let t2 = dmul(arena, a, db);
                dadd(arena, t1, t2)
            }
            // Quotient rule: (a/b)' = (a'·b - a·b') / b².
            OpKind::Div => {
                let da = differentiate(arena, a, var, depth + 1, memo)?;
                let db = differentiate(arena, b, var, depth + 1, memo)?;
                let t1 = dmul(arena, da, b);
                let t2 = dmul(arena, a, db);
                let num = dsub(arena, t1, t2);
                let den = dmul(arena, b, b);
                ddiv(arena, num, den)
            }
            // min/max are piecewise: the active branch's derivative, selected
            // on the same comparison Jet2 uses (Lt for min, Gt for max — ties
            // take the right operand's derivative).
            OpKind::Min | OpKind::Max => {
                let da = differentiate(arena, a, var, depth + 1, memo)?;
                let db = differentiate(arena, b, var, depth + 1, memo)?;
                let cmp = if op == OpKind::Min {
                    OpKind::Lt
                } else {
                    OpKind::Gt
                };
                let mask = arena.push_binary(cmp, a, b);
                select_branch(arena, mask, da, db)
            }
            // (a^b)' = a^b · (b'·ln a + b·a'/a). NaN for a ≤ 0 with varying b,
            // exactly as the analytic derivative is.
            OpKind::Pow => {
                let da = differentiate(arena, a, var, depth + 1, memo)?;
                let db = differentiate(arena, b, var, depth + 1, memo)?;
                let ln_a = arena.push_unary(OpKind::Ln, a);
                let t1 = dmul(arena, db, ln_a);
                let ratio = ddiv(arena, da, a);
                let t2 = dmul(arena, b, ratio);
                let sum = dadd(arena, t1, t2);
                let p = arena.push_binary(OpKind::Pow, a, b);
                dmul(arena, p, sum)
            }
            // hypot(a,b)' = (a·a' + b·b') / hypot(a,b).
            OpKind::Hypot => {
                let da = differentiate(arena, a, var, depth + 1, memo)?;
                let db = differentiate(arena, b, var, depth + 1, memo)?;
                let t1 = dmul(arena, a, da);
                let t2 = dmul(arena, b, db);
                let num = dadd(arena, t1, t2);
                let h = arena.push_binary(OpKind::Hypot, a, b);
                ddiv(arena, num, h)
            }
            // atan2(y,x)' = (x·y' - y·x') / (x² + y²).
            OpKind::Atan2 => {
                let dy = differentiate(arena, a, var, depth + 1, memo)?;
                let dx = differentiate(arena, b, var, depth + 1, memo)?;
                let t1 = dmul(arena, b, dy);
                let t2 = dmul(arena, a, dx);
                let num = dsub(arena, t1, t2);
                let y2 = dmul(arena, a, a);
                let x2 = dmul(arena, b, b);
                let den = dadd(arena, x2, y2);
                ddiv(arena, num, den)
            }
            // Comparison masks are step functions: derivative 0 a.e.
            OpKind::Lt | OpKind::Le | OpKind::Gt | OpKind::Ge | OpKind::Eq | OpKind::Ne => {
                arena.push_const(0.0)
            }
            _ => return Err(LowerError::UnsupportedOp { op }),
        },

        ExprNode::Ternary(op, a, b, c) => match op {
            // (a·b + c)' = a'·b + a·b' + c'.
            OpKind::MulAdd => {
                let da = differentiate(arena, a, var, depth + 1, memo)?;
                let db = differentiate(arena, b, var, depth + 1, memo)?;
                let dc = differentiate(arena, c, var, depth + 1, memo)?;
                let t1 = dmul(arena, da, b);
                let t2 = dmul(arena, a, db);
                let prod = dadd(arena, t1, t2);
                dadd(arena, prod, dc)
            }
            // Select is a branch: the derivative follows the taken branch.
            // The condition itself is a step function (derivative 0).
            OpKind::Select => {
                let db = differentiate(arena, b, var, depth + 1, memo)?;
                let dc = differentiate(arena, c, var, depth + 1, memo)?;
                select_branch(arena, a, db, dc)
            }
            // clamp(x, lo, hi) is piecewise: lo' below, hi' above, x' between
            // (Jet2's convention: Lt-low then Gt-high).
            OpKind::Clamp => {
                let dx = differentiate(arena, a, var, depth + 1, memo)?;
                let dlo = differentiate(arena, b, var, depth + 1, memo)?;
                let dhi = differentiate(arena, c, var, depth + 1, memo)?;
                let below = arena.push_binary(OpKind::Lt, a, b);
                let above = arena.push_binary(OpKind::Gt, a, c);
                let upper = select_branch(arena, above, dhi, dx);
                select_branch(arena, below, dlo, upper)
            }
            // Buffer reads are piecewise constant in their indices: 0. See
            // `lower_dwrt` docs.
            OpKind::Gather => arena.push_const(0.0),
            _ => return Err(LowerError::UnsupportedOp { op }),
        },

        ExprNode::Nary(op, _, _) => return Err(LowerError::UnsupportedOp { op }),
    };

    memo.insert((id.0, var), d);
    Ok(d)
}

/// `Select(mask, a, b)`, folded to `a` when both branches are the same node
/// (in particular when both derivatives are statically zero).
fn select_branch(arena: &mut ExprArena, mask: ExprId, a: ExprId, b: ExprId) -> ExprId {
    if a == b || (is_const_zero(arena, a) && is_const_zero(arena, b)) {
        return a;
    }
    arena.push_ternary(OpKind::Select, mask, a, b)
}

/// `√(1 - u²)` — shared denominator of the asin/acos rules.
fn one_minus_sq_sqrt(arena: &mut ExprArena, u: ExprId) -> ExprId {
    let one = arena.push_const(1.0);
    let u2 = dmul(arena, u, u);
    let t = dsub(arena, one, u2);
    arena.push_unary(OpKind::Sqrt, t)
}
