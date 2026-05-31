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

/// Whether `op` is a transcendental this pass expands (so backends never see it).
fn is_transcendental(op: OpKind) -> bool {
    matches!(
        op,
        OpKind::Sin | OpKind::Cos | OpKind::Tan
    )
}

/// Expand every transcendental node reachable from `root` into a primitive
/// arithmetic subgraph, returning the (possibly new) root in the same arena.
///
/// Post-order rebuild over the arena, mirroring [`ExprArena::substitute_params`]:
/// children are lowered first, then each node is re-emitted with its children
/// remapped — except transcendental nodes, which are replaced by their
/// polynomial expansion. Shared subexpressions are lowered once (the `id_map`
/// dedups), so the DAG structure is preserved.
pub fn expand_transcendentals(arena: &mut ExprArena, root: ExprId) -> ExprId {
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
                match arena.node(id).clone() {
                    ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Param(_) => {}
                    ExprNode::Unary(_, a) => work.push(Task::Descend(a)),
                    ExprNode::Binary(_, a, b) => {
                        work.push(Task::Descend(b));
                        work.push(Task::Descend(a));
                    }
                    ExprNode::Ternary(_, a, b, c) => {
                        work.push(Task::Descend(c));
                        work.push(Task::Descend(b));
                        work.push(Task::Descend(a));
                    }
                    ExprNode::Nary(_, start, len) => {
                        let (s, l) = (start as usize, len as usize);
                        let children: Vec<ExprId> =
                            arena.nary_children_raw()[s..s + l].to_vec();
                        for child in children.into_iter().rev() {
                            work.push(Task::Descend(child));
                        }
                    }
                }
            }
            Task::Emit(id) => {
                if id_map[id.0 as usize].is_some() {
                    continue;
                }
                let m = |old: ExprId| id_map[old.0 as usize].expect("child lowered before parent");
                let new_id = match arena.node(id).clone() {
                    ExprNode::Var(i) => arena.push_var(i),
                    ExprNode::Const(v) => arena.push_const(v),
                    ExprNode::Param(i) => arena.push_param(i),
                    ExprNode::Unary(op, a) => {
                        let a = m(a);
                        if is_transcendental(op) {
                            expand_unary(arena, op, a)
                        } else {
                            arena.push_unary(op, a)
                        }
                    }
                    ExprNode::Binary(op, a, b) => arena.push_binary(op, m(a), m(b)),
                    ExprNode::Ternary(op, a, b, c) => arena.push_ternary(op, m(a), m(b), m(c)),
                    ExprNode::Nary(op, start, len) => {
                        let (s, l) = (start as usize, len as usize);
                        let mapped: Vec<ExprId> = arena.nary_children_raw()[s..s + l]
                            .to_vec()
                            .into_iter()
                            .map(m)
                            .collect();
                        arena.push_nary(op, &mapped)
                    }
                };
                id_map[id.0 as usize] = Some(new_id);
            }
        }
    }

    id_map[root.0 as usize].expect("root lowered")
}

/// Convenience wrapper for the public `compile_arena_dag*` entries, which hold a
/// shared `&ExprArena`: clone it, expand transcendentals in the clone, and
/// return the owned arena + new root. Cheap when there are no transcendentals
/// (the clone is two `Vec`s and the walk just copies), so every entry can call
/// it unconditionally and be sure no backend — per-batch, scanline, or jet —
/// ever sees a transcendental node.
pub fn expand_transcendentals_owned(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    // Identity fast-path: if there is nothing to lower, return the arena
    // unchanged. The rebuild below is not bit-identical to the input (it can
    // re-order / re-dedup nodes), which would perturb register allocation for
    // transcendental-free kernels; skipping it keeps lowering a true no-op for
    // them.
    if !arena
        .nodes_raw()
        .iter()
        .any(|n| matches!(n, ExprNode::Unary(op, _) if is_transcendental(*op)))
    {
        return (arena.clone(), root);
    }
    let mut owned = arena.clone();
    let new_root = expand_transcendentals(&mut owned, root);
    (owned, new_root)
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
        _ => unreachable!("expand_unary called on non-transcendental {op:?}"),
    }
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
    let c1 = arena.push_const(3.141_592_653_589_79);
    let c3 = arena.push_const(-5.167_712_780_049_97);
    let c5 = arena.push_const(2.550_164_039_877_34);
    let c7 = arena.push_const(-0.599_264_528_932_149);
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
