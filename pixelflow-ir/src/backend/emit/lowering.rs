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
                let new_id = match lower(arena, &node, &m) {
                    Some(new) => new,
                    None => copy_node(arena, &node, &m),
                };
                id_map[id.0 as usize] = Some(new_id);
            }
        }
    }

    id_map[root.0 as usize].expect("root lowered")
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
    let c0 = arena.push_const(3.333_333_1e-1);
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
