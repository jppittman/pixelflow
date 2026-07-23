//! Global JIT compile cache: identical kernels compile once.
//!
//! P0 (docs/results/2026-07-20-jit-compile-cost.md) showed codegen — not
//! executable-memory management — is the compile-cost floor, so the lever
//! that actually pays is skipping codegen entirely for a kernel that was
//! already compiled. The macro-emitted builders hit this constantly: every
//! call of an N-param builder with the same arguments (window resizes), and
//! every structurally identical glyph kernel in a bake sweep, produces a
//! byte-identical schedule.
//!
//! Keys are the **canonical form of the reachable subgraph**: nodes in
//! ascending id order with ids remapped dense. Construction garbage (dead
//! nodes left behind by `substitute_params` / splicing rebuilds) does not
//! perturb the key, so logically identical kernels hit regardless of build
//! history. Keys are compared by full equality — a hash collision can cause
//! a wasted probe, never wrong code.
//!
//! Kernels that read bound memory (`Buffer` leaves) are compiled fresh: their
//! ABI differs (context-pointer calls) and their code bakes buffer slot
//! metadata. No kernel-macro surface produces them today.
//!
//! The cache is unbounded: entries are one executable-memory region each and
//! the population is the program's distinct kernel set, which is bounded by
//! construction (kernels are made at load/composition time, not per frame).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::vec::Vec;

use crate::JitManifold;
use crate::arena::{ExprArena, ExprId, ExprNode};
use crate::backend::emit;

static CACHE: OnceLock<Mutex<HashMap<Vec<u8>, Arc<JitManifold>>>> = OnceLock::new();

/// Compile the kernel rooted at `root`, sharing previously compiled code for
/// canonically identical kernels. The returned `Arc` is the shared handle —
/// two constructions of the same kernel yield pointer-equal manifolds.
pub fn compile_cached(
    arena: &ExprArena,
    root: ExprId,
) -> Result<Arc<JitManifold>, &'static str> {
    let Some(key) = canonical_key(arena, root) else {
        // Uncacheable (bound memory): compile fresh.
        let result = emit::compile_arena_dag(arena, root)?;
        return Ok(Arc::new(JitManifold::new(result.code)));
    };

    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache
        .lock()
        .expect("jit_cache: lock poisoned")
        .get(&key)
    {
        return Ok(hit.clone());
    }

    // Compile outside the lock so concurrent distinct-kernel constructions
    // don't serialize. A racing duplicate compile wastes work; the first
    // insertion wins so all callers share one region.
    let result = emit::compile_arena_dag(arena, root)?;
    let compiled = Arc::new(JitManifold::new(result.code));
    let mut guard = cache.lock().expect("jit_cache: lock poisoned");
    Ok(guard.entry(key).or_insert(compiled).clone())
}

/// Number of distinct kernels interned so far (test/telemetry hook).
#[must_use]
pub fn entry_count() -> usize {
    CACHE
        .get()
        .map(|c| c.lock().expect("jit_cache: lock poisoned").len())
        .unwrap_or(0)
}

/// Canonical serialization of the subgraph reachable from `root`: nodes in
/// ascending original id order (the arena is append-only, so children always
/// precede parents), child references remapped to dense indices. `None` if
/// the subgraph reads bound memory.
fn canonical_key(arena: &ExprArena, root: ExprId) -> Option<Vec<u8>> {
    let len = arena.nodes_raw().len();
    let mut reachable = vec![false; len];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if core::mem::replace(&mut reachable[id.0 as usize], true) {
            continue;
        }
        stack.extend(arena.children(id));
    }

    // Dense remap in ascending id order.
    let mut dense: Vec<u32> = vec![u32::MAX; len];
    let mut next = 0u32;
    let mut key: Vec<u8> = Vec::with_capacity(len * 8);

    let push_id = |key: &mut Vec<u8>, dense: &[u32], id: ExprId| {
        let d = dense[id.0 as usize];
        debug_assert_ne!(d, u32::MAX, "child densified before parent");
        key.extend_from_slice(&d.to_le_bytes());
    };

    for idx in 0..len {
        if !reachable[idx] {
            continue;
        }
        match arena.node(ExprId(idx as u32)) {
            ExprNode::Var(i) => {
                key.push(0);
                key.push(*i);
            }
            ExprNode::Const(v) => {
                key.push(1);
                key.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            ExprNode::Param(i) => {
                key.push(2);
                key.push(*i);
            }
            ExprNode::Buffer(_) => return None,
            ExprNode::Unary(op, a) => {
                key.push(3);
                key.push(*op as u8);
                push_id(&mut key, &dense, *a);
            }
            ExprNode::Binary(op, a, b) => {
                key.push(4);
                key.push(*op as u8);
                push_id(&mut key, &dense, *a);
                push_id(&mut key, &dense, *b);
            }
            ExprNode::Ternary(op, a, b, c) => {
                key.push(5);
                key.push(*op as u8);
                push_id(&mut key, &dense, *a);
                push_id(&mut key, &dense, *b);
                push_id(&mut key, &dense, *c);
            }
            ExprNode::Nary(op, start, n) => {
                key.push(6);
                key.push(*op as u8);
                key.extend_from_slice(&n.to_le_bytes());
                let (s, l) = (*start as usize, *n as usize);
                for child in &arena.nary_children_raw()[s..s + l] {
                    push_id(&mut key, &dense, *child);
                }
            }
        }
        dense[idx] = next;
        next += 1;
    }

    Some(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kind::OpKind;

    fn circle_arena(garbage: bool) -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        if garbage {
            // Construction garbage: unreachable nodes must not perturb the key.
            let g = a.push_const(123.0);
            let _ = a.push_unary(OpKind::Sqrt, g);
        }
        let x = a.push_var(0);
        let y = a.push_var(1);
        let x2 = a.push_binary(OpKind::Mul, x, x);
        let y2 = a.push_binary(OpKind::Mul, y, y);
        let s = a.push_binary(OpKind::Add, x2, y2);
        let root = a.push_unary(OpKind::Sqrt, s);
        (a, root)
    }

    #[test]
    fn identical_kernels_share_code() {
        let (a1, r1) = circle_arena(false);
        let (a2, r2) = circle_arena(true);
        let m1 = compile_cached(&a1, r1).expect("compile");
        let m2 = compile_cached(&a2, r2).expect("compile");
        assert!(
            Arc::ptr_eq(&m1, &m2),
            "canonically identical kernels must share one compiled region"
        );
    }

    #[test]
    fn distinct_kernels_do_not_collide() {
        let (a1, r1) = circle_arena(false);
        let mut a2 = ExprArena::new();
        let x = a2.push_var(0);
        let y = a2.push_var(1);
        let r2 = a2.push_binary(OpKind::Sub, x, y);
        let m1 = compile_cached(&a1, r1).expect("compile");
        let m2 = compile_cached(&a2, r2).expect("compile");
        assert!(!Arc::ptr_eq(&m1, &m2));
    }

    #[test]
    fn key_is_garbage_insensitive_and_structure_sensitive() {
        let (a1, r1) = circle_arena(false);
        let (a2, r2) = circle_arena(true);
        assert_eq!(canonical_key(&a1, r1), canonical_key(&a2, r2));

        let mut a3 = ExprArena::new();
        let x = a3.push_var(0);
        let y = a3.push_var(1);
        let x2 = a3.push_binary(OpKind::Mul, x, x);
        let y2 = a3.push_binary(OpKind::Mul, y, y);
        let s = a3.push_binary(OpKind::Add, y2, x2); // operand order flipped
        let r3 = a3.push_unary(OpKind::Sqrt, s);
        assert_ne!(canonical_key(&a1, r1), canonical_key(&a3, r3));
    }
}
