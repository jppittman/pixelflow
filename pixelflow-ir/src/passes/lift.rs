//! The "unit" lowering of `Apply`: a name read at a warp, held as a leaf.
//!
//! `Apply(k, cx, cy)` is not a unit — a unit is a closed term read at its own
//! coordinates, and the substitution an `Apply` stands for cannot cross the
//! boundary a unit is optimized behind (the runtime tier's Law U). But the
//! *composite* `k ∘ (cx, cy)` is closed whenever the warp is, so it can be
//! named — interned as a kernel of its own — and the term around it can hold
//! that name as an opaque leaf, exactly as it holds a `Ref`.
//!
//! What the leaf stands for after extraction is not the composite's own
//! optimization but `Apply(k, cx, cy)` again: the runtime tier optimizes `k`
//! once, however many warps read it, and [`link`](super::link) resolves each
//! `Apply` over that one optimized body. So `K` copies of one referent cost
//! one saturation, and each copy gets the referent's whole budget, instead of
//! all `K` sharing the budget of the one term that inlined them
//! (docs/language/2026-09-09-composition-is-linking.md §9.2).
//!
//! What is not lifted is left an `Apply` for [`expand_applies`](super::expand_applies):
//! a warp that reads a binder, a uniform, a buffer or a name, or a referent
//! that declares a uniform or buffer, whose slots a bare composite kernel could
//! not carry.

use super::{expand_applies, rebuild_arena};
use crate::arena::{ExprArena, ExprId, ExprNode};
use crate::kernel::Kernel;
use crate::key::KernelKey;
use crate::store::KernelStore;
use crate::variance::{Variance, compute_arena_variance};

/// An `Apply` held as a leaf.
pub struct Lifted {
    /// The name of the composite, which the term around holds as a `Ref`.
    pub leaf: KernelKey,
    /// What the leaf links to after extraction: a term whose root is the
    /// `Apply`, over the warp spliced in beside it.
    pub apply: (ExprArena, ExprId),
}

/// Replace every liftable [`ExprNode::Apply`] reachable from `root` with a
/// `Ref` to its composite, returning the new root and what each leaf stands
/// for. An `Apply` that cannot be lifted stays one.
pub fn lift_applies(arena: &mut ExprArena, root: ExprId) -> (ExprId, Vec<Lifted>) {
    let mut lifted = Vec::new();
    let new_root = rebuild_arena(arena, root, |arena, node, m| {
        let ExprNode::Apply(key, cx, cy) = *node else {
            return None;
        };
        let referent = KernelStore::resolve(key)?;
        if !referent.uniforms().is_empty() || !referent.buffers().is_empty() {
            return None;
        }
        let (mut apply, wx, wy) = closed_warp(arena, m(cx), m(cy))?;
        let apply_root = apply.push_apply(key, wx, wy);
        let mut composite = apply.clone();
        let composite_root = expand_applies(&mut composite, apply_root);
        let leaf = KernelStore::intern(&Kernel::from_parts(composite, composite_root));
        lifted.push(Lifted {
            leaf,
            apply: (apply, apply_root),
        });
        Some(arena.push_ref(leaf))
    });
    (new_root, lifted)
}

/// The warps `cx`, `cy` of `arena` spliced into a fresh arena, and their roots
/// there — `None` unless each is a term over the coordinates and literals
/// only.
fn closed_warp(arena: &ExprArena, cx: ExprId, cy: ExprId) -> Option<(ExprArena, ExprId, ExprId)> {
    let mut warp = ExprArena::new();
    let wx = warp.splice(arena, cx);
    let wy = warp.splice(arena, cy);
    let variance = compute_arena_variance(&warp);
    let plain = !warp.names_a_kernel()
        && warp.uniforms().is_empty()
        && warp.buffers().is_empty()
        && [wx, wy].into_iter().all(|w| {
            warp.free_index(w).is_none()
                && variance[w.0 as usize].without(Variance::COORDS).is_const()
        });
    plain.then_some((warp, wx, wy))
}
