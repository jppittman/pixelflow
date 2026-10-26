//! Global JIT compile cache: identical kernels compile once.
//!
//! Skipping codegen entirely avoids recompiling a kernel that was already
//! compiled. The macro-emitted builders can hit this frequently: every
//! call of an N-param builder with the same arguments (window resizes), and
//! every structurally identical glyph kernel in a bake sweep, produces a
//! byte-identical schedule.
//!
//! The key is [`pixelflow_ir::key::canonical`]'s — a kernel's identity is not
//! codegen's private business, so it lives at the bottom of the dependency
//! graph where a `Ref` can name into it too
//! (docs/language/2026-09-09-composition-is-linking.md §2). This cache is one
//! consumer of it.
//!
//! Keys are the [`LatticeShape`] the kernel is compiled for — its extents,
//! so a lattice of a different size is a different kernel and a window
//! resize recompiles, by decision — plus the **canonical form of the
//! reachable subgraph**: a post-order walk from the root with structurally
//! equal subterms hash-consed, so neither construction garbage (dead nodes
//! left behind by `substitute_vars_with` / splicing rebuilds) nor the order a
//! builder pushed the live nodes in perturbs the key, and logically
//! identical kernels hit regardless of build history. Keys are compared by
//! full equality — a hash collision can cause a wasted probe, never wrong
//! code.
//!
//! ## The link step
//!
//! A kernel that reads a buffer or a uniform names it by *identity*, and
//! identities are minted per instance — so keyed by identity, two
//! compositions of the same shape would never share code. They share it
//! instead by **dense slot**: the same canonical traversal that produces the
//! key numbers each distinct buffer and uniform identity by first
//! occurrence, and that numbering is what the emitted code is compiled
//! against — the buffer's context entry, the uniform's offset in the block.
//! The traversal is a function of structure alone, so the same shape gets
//! the same numbering, and a thousand circles differing only in which
//! instances they read are one compile. What differs between them is the
//! [`Linked`] table handed back with the code: which identity each slot
//! binds, which the caller uses to build the context and the block.
//!
//! Buffer *extents* stay in the key. Two kernels over buffers of different
//! shapes are different code, since the shape is what the address arithmetic
//! was folded against.
//!
//! The cache is unbounded: entries are one executable-memory region each and
//! the population is the program's distinct kernel set, which is bounded by
//! construction (kernels are made at load/composition time, not per frame).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::vec::Vec;

use crate::emit;
use crate::emit::CompiledKernel;
use crate::error::CompileError;
use pixelflow_ir::LatticeShape;
use pixelflow_ir::arena::{BufferDecl, ExprNode, UniformDecl};
use pixelflow_ir::key::{Canonical, canonical};

static CACHE: OnceLock<Mutex<HashMap<CacheKey, Arc<CompiledKernel>>>> = OnceLock::new();

/// Compiled code plus the link: which identity each slot the code was
/// compiled against binds.
///
/// The code is shared through the cache with every kernel of the same shape;
/// the link is this kernel's own. A caller builds the context from
/// `buffers` in slot order, and the block from `uniforms` in offset order —
/// the block pointer goes in the context entry after the last buffer, and
/// only when `uniforms` is non-empty.
pub struct Linked {
    /// The shared code.
    pub kernel: Arc<CompiledKernel>,
    /// The buffer each context slot binds, in slot order.
    pub buffers: Vec<BufferDecl>,
    /// The uniform each block offset holds, in offset order.
    pub uniforms: Vec<UniformDecl>,
}

/// Compile a [`Kernel`](pixelflow_ir::Kernel) to an executable [`CompiledKernel`]
/// for a lattice of the given `shape`, sharing previously compiled code for
/// canonically identical kernels at the same extents.
///
/// The returned `Arc` is the shared handle — two constructions of the same
/// kernel at the same shape yield pointer-equal manifolds — and the link
/// beside it says what this construction's slots bind. A fold whose bound is a
/// value compiles inside its guard (`Kernel::over_values`), and every count it
/// is run with shares the one program.
///
/// # Errors
///
/// Whatever [`emit::compile`] reports for the (optimized) arena.
///
/// # Panics
///
/// Panics if the arena — as handed in, or as saturation leaves it — names a
/// retired coordinate axis (`Var(2)`/`Var(3)`, the old Z and W), or if `shape`
/// has a zero extent, which has no sample to compute, or if a fold's bound is
/// a mask an emitter holds as one, which is refused by name (`BoundSort`) at
/// either end on every tier, since a mask names no count.
/// `Kernel::over_values` takes no such bound, but a kernel adopted with
/// `Kernel::from_parts` can carry one. The assertions live one layer down, in
/// [`emit::compile`], because that is the boundary every route to machine code
/// passes through and this is only one of them.
pub fn compile(kernel: &pixelflow_ir::Kernel, shape: LatticeShape) -> Result<Linked, CompileError> {
    let (arena, root) = kernel.parts();
    // The link tables, and the key, are read off the kernel with every
    // reference expanded. A `Ref` is a leaf whose body — and whose buffer and
    // uniform declarations — are not in this arena, so a key taken off the
    // arena alone would name a kernel other than the one that gets emitted,
    // and the link handed back would be missing every slot the referent
    // reads. Slot order is ABI, and a unit's uniforms are slots.
    //
    // Guarded rather than called unconditionally: the pass's own identity
    // path still clones the arena, and this runs on every compile including
    // the cache hits.
    let holds_a_ref = arena.nodes().any(|(_, n)| matches!(n, ExprNode::Ref(_)));
    let expanded = holds_a_ref.then(|| pixelflow_ir::passes::expand_refs_owned(arena, root));
    let (linked_arena, linked_root) = match &expanded {
        Some((linked, linked_root)) => (linked, *linked_root),
        None => (arena, root),
    };
    let Canonical {
        key: expanded_key,
        buffers,
        uniforms,
    } = canonical(linked_arena, linked_root);

    // Optimize, link, then emit. This is not a step callers get to sequence:
    // an arena reaching a backend unoptimized is never what anyone wanted, so
    // it is inside the compile entry, the one place it cannot be forgotten.
    //
    // The optimizer is handed the arena *as written*, references and all: a
    // reference is a unit, optimized by itself and linked back in after
    // extraction (`pixelflow_search::runtime`'s units;
    // docs/language/2026-09-25-the-language-is-kernel.md §4, O1).
    //
    // It bails for constructs the e-graph does not model
    // (`egraph::insert`'s `Declined`: a `Param`, an op
    // `Vocabulary::Runtime` does not admit, such as a `RawGather`) in a
    // term with no unit; those still compile — from the *expanded* arena, so
    // a reference never reaches the emitter — just without the extra fusion,
    // and saturation telemetry records the decline. A `Reduce` is modelled —
    // it enters the e-graph as itself, and a fold extraction keeps is
    // emitted as a loop.
    //
    // The relink after it renumbers the tables into the canonical order the
    // key was built from — extraction redeclares identities in its own
    // walk order — without touching a node, so the bytes of a kernel that
    // declares neither a buffer nor a uniform are exactly what they were.
    let emit_fn = || {
        let optimized = pixelflow_search::runtime::optimize_runtime_arena(arena, root, shape);
        let (arena, root) = optimized
            .as_deref()
            .map(|(a, r)| (a, *r))
            .unwrap_or((linked_arena, linked_root));
        if buffers.is_empty() && uniforms.is_empty() {
            return emit::compile(arena, root, shape);
        }
        let (linked, root) = arena.relink(root, &buffers, &uniforms);
        emit::compile(&linked, root, shape)
    };

    // Keyed on the arena *as handed in*, before optimization, plus the shape.
    // Optimization is a deterministic function of those two, so equal inputs
    // yield equal output and a hit skips the saturation as well as the codegen.
    let key = CacheKey {
        expanded: expanded_key,
        shape,
        as_written: holds_a_ref.then(|| canonical(arena, root).key),
    };
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache.lock().expect("jit_cache: lock poisoned").get(&key) {
        return Ok(Linked {
            kernel: hit.clone(),
            buffers,
            uniforms,
        });
    }

    // Compile outside the lock so concurrent distinct-kernel constructions
    // don't serialize. A racing duplicate compile wastes work; the first
    // insertion wins so all callers share one region.
    let result = emit_fn()?;
    let compiled = Arc::new(result.code);
    let mut guard = cache.lock().expect("jit_cache: lock poisoned");
    let kernel = guard.entry(key).or_insert(compiled).clone();
    Ok(Linked {
        kernel,
        buffers,
        uniforms,
    })
}

/// The compile cache's key: what the code is a function of.
///
/// Without units that is the expanded kernel's structure and the shape.
/// **With units it is also where they are.** The expanded structure cannot
/// say: `body` and `body.by_ref()` expand alike and are different programs,
/// since a unit is optimized by itself, so on that key whichever compiled
/// first would answer for both and the bytes would depend on compile order.
/// `as_written` — the kernel's own canonical bytes, in which a `Ref` is its
/// referent's identity — says it, and is `None` for a kernel that names no
/// unit.
///
/// The cost of keying on identity, recorded in
/// docs/language/2026-09-25-the-language-is-kernel.md §4 O1: a unit's identity
/// digests the uniforms it was minted with, so a font rebuilt over fresh
/// uniforms is a new entry here, where the same structure without units
/// would hit. A structural key that walks through the units is the caching
/// JP has deferred.
#[derive(PartialEq, Eq, Hash)]
struct CacheKey {
    expanded: Vec<u8>,
    shape: LatticeShape,
    as_written: Option<Vec<u8>>,
}

/// Number of distinct kernels interned so far (test/telemetry hook).
#[must_use]
pub fn entry_count() -> usize {
    CACHE
        .get()
        .map(|c| c.lock().expect("jit_cache: lock poisoned").len())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::Kernel;
    use pixelflow_ir::arena::{BufferIdentity, ExprArena, UniformIdentity};
    use pixelflow_ir::fold::{Binder, Bound, Fold, Monoid};
    use pixelflow_ir::kind::OpKind;

    const TEST_SHAPE: LatticeShape = LatticeShape::new([64, 64]);

    fn kernel_of(kernel: &Kernel) -> Arc<CompiledKernel> {
        compile(kernel, TEST_SHAPE).expect("compile").kernel
    }

    fn circle_arena() -> Kernel {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let x2 = a.push_binary(OpKind::Mul, x, x);
        let y2 = a.push_binary(OpKind::Mul, y, y);
        let s = a.push_binary(OpKind::Add, x2, y2);
        let root = a.push_unary(OpKind::Sqrt, s);
        Kernel::from_parts(a, root)
    }

    #[test]
    fn distinct_kernels_do_not_collide() {
        let k1 = circle_arena();
        let mut a2 = ExprArena::new();
        let x = a2.push_var(0);
        let y = a2.push_var(1);
        let r2 = a2.push_binary(OpKind::Sub, x, y);
        let k2 = Kernel::from_parts(a2, r2);
        let m1 = kernel_of(&k1);
        let m2 = kernel_of(&k2);
        assert!(!Arc::ptr_eq(&m1, &m2));
    }

    #[test]
    fn a_folds_range_is_part_of_its_kernel_identity() {
        // Two kernels alike but for one fold's extent must not share a cache
        // entry: the property is that `FoldHeader::to_bits` is in the key.
        let slot = |s: u8| Binder::from_slot(s).expect("a live binder");
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let body1 = a.push_binary(OpKind::Add, x, x);
        let r1 = a.push_reduce(
            Fold::new(Monoid::SUM, slot(0)),
            Bound::Lit(0),
            Bound::Lit(3),
            body1,
        );
        let body2 = a.push_binary(OpKind::Mul, x, x);
        let r2 = a.push_reduce(
            Fold::new(Monoid::PRODUCT, slot(1)),
            Bound::Lit(0),
            Bound::Lit(6),
            body2,
        );
        let root = a.push_binary(OpKind::Add, r1, r2);

        let mut a2 = ExprArena::new();
        let x2 = a2.push_var(0);
        let body1b = a2.push_binary(OpKind::Add, x2, x2);
        let r1b = a2.push_reduce(
            Fold::new(Monoid::SUM, slot(0)),
            Bound::Lit(0),
            Bound::Lit(3),
            body1b,
        );
        let body2b = a2.push_binary(OpKind::Mul, x2, x2);
        // The extent differs only here.
        let r2b = a2.push_reduce(
            Fold::new(Monoid::PRODUCT, slot(1)),
            Bound::Lit(0),
            Bound::Lit(9),
            body2b,
        );
        let root2 = a2.push_binary(OpKind::Add, r1b, r2b);

        let m1 = kernel_of(&Kernel::from_parts(a, root));
        let m2 = kernel_of(&Kernel::from_parts(a2, root2));
        assert!(
            !Arc::ptr_eq(&m1, &m2),
            "a change confined to the second reduce's extent must not share a cache entry"
        );
    }

    /// A subterm written twice and the same subterm shared are one program,
    /// so they are one compiled region — and one argument: the link folds
    /// the two slots naming one instance back to one, which is the only
    /// duplicate an interning arena can carry.
    #[test]
    fn a_duplicated_subterm_and_a_shared_one_are_one_entry() {
        let decl = UniformDecl {
            id: UniformIdentity::mint(),
            default: 0.25,
        };
        let shared = {
            let mut a = ExprArena::new();
            let u = a.declare_uniform(decl);
            let x = a.push_var(0);
            let leaf = a.push_uniform(u);
            let xu = a.push_binary(OpKind::Mul, x, leaf);
            let root = a.push_binary(OpKind::Add, xu, xu);
            Kernel::from_parts(a, root)
        };
        let duplicated = {
            let mut a = ExprArena::new();
            let u1 = a.declare_uniform(decl);
            let u2 = a.declare_uniform(decl);
            let x = a.push_var(0);
            let leaf1 = a.push_uniform(u1);
            let leaf2 = a.push_uniform(u2);
            let xu1 = a.push_binary(OpKind::Mul, x, leaf1);
            let xu2 = a.push_binary(OpKind::Mul, x, leaf2);
            let root = a.push_binary(OpKind::Add, xu1, xu2);
            Kernel::from_parts(a, root)
        };
        let l_shared = compile(&shared, TEST_SHAPE).expect("compile");
        let l_duplicated = compile(&duplicated, TEST_SHAPE).expect("compile");
        assert!(
            Arc::ptr_eq(&l_shared.kernel, &l_duplicated.kernel),
            "a duplicated subterm and a shared one must share one compiled region"
        );
        assert_eq!(l_shared.uniforms, [decl]);
        assert_eq!(l_duplicated.uniforms, [decl], "two slots, one argument");
    }

    /// One argument read twice and two arguments read once each are two
    /// programs: the same shape bytes but for the slot numbers, which is
    /// enough.
    #[test]
    fn one_argument_read_twice_and_two_arguments_are_two_entries() {
        let decl = |default| UniformDecl {
            id: UniformIdentity::mint(),
            default,
        };
        let one_argument = {
            let mut a = ExprArena::new();
            let u = a.declare_uniform(decl(0.0));
            let leaf = a.push_uniform(u);
            let root = a.push_binary(OpKind::Add, leaf, leaf);
            Kernel::from_parts(a, root)
        };
        let two_arguments = {
            let mut a = ExprArena::new();
            let u = a.declare_uniform(decl(0.0));
            let v = a.declare_uniform(decl(0.0));
            let (lu, lv) = (a.push_uniform(u), a.push_uniform(v));
            let root = a.push_binary(OpKind::Add, lu, lv);
            Kernel::from_parts(a, root)
        };
        assert!(
            !Arc::ptr_eq(&kernel_of(&one_argument), &kernel_of(&two_arguments)),
            "u + u and u + v must not share a cache entry"
        );
    }

    #[test]
    fn same_kernel_at_two_extents_is_two_entries() {
        let k = circle_arena();
        let frame = kernel_of(&k);
        let again = kernel_of(&k);
        let wider = compile(&k, LatticeShape::new([65, 64]))
            .expect("compile")
            .kernel;
        assert!(
            Arc::ptr_eq(&frame, &again),
            "the same kernel at the same extents must share one entry"
        );
        assert!(
            !Arc::ptr_eq(&frame, &wider),
            "one more column is a different lattice, hence a different kernel"
        );
    }

    // ─────────────────────── the link step ───────────────────────

    /// `(x − cx)·r + cy` over fresh uniform instances: one shape, many
    /// factors. `declared_first` flips the table order so the link, not the
    /// declaration order, is what the code is compiled against.
    /// `canonical` over a whole kernel — its fragment is already linked in
    /// these tests, so the arena and root are just its parts.
    fn canon(k: &Kernel) -> Canonical {
        let (arena, root) = k.parts();
        canonical(arena, root)
    }

    fn circle_of(declared_first: bool) -> (Kernel, [UniformDecl; 3]) {
        let decl = |default| UniformDecl {
            id: UniformIdentity::mint(),
            default,
        };
        let (cx, cy, r) = (decl(0.0), decl(0.0), decl(1.0));
        let mut a = ExprArena::new();
        let (scx, scy, sr) = if declared_first {
            let scx = a.declare_uniform(cx);
            let scy = a.declare_uniform(cy);
            let sr = a.declare_uniform(r);
            (scx, scy, sr)
        } else {
            let sr = a.declare_uniform(r);
            let scy = a.declare_uniform(cy);
            let scx = a.declare_uniform(cx);
            (scx, scy, sr)
        };
        let x = a.push_var(0);
        let ucx = a.push_uniform(scx);
        let ur = a.push_uniform(sr);
        let ucy = a.push_uniform(scy);
        let d = a.push_binary(OpKind::Sub, x, ucx);
        let scaled = a.push_binary(OpKind::Mul, d, ur);
        let root = a.push_binary(OpKind::Add, scaled, ucy);
        (Kernel::from_parts(a, root), [cx, r, cy])
    }

    /// Every instance shares the one region and gets its own link.
    #[test]
    fn a_thousand_circles_share_one_region_with_a_thousand_links() {
        let mut first: Option<Arc<CompiledKernel>> = None;
        for i in 0..1000 {
            let (k, [cx, r, cy]) = circle_of(i % 2 == 0);
            let linked = compile(&k, TEST_SHAPE).expect("compile");
            // The link is this instance's, in first-occurrence order.
            assert_eq!(linked.uniforms, [cx, r, cy]);
            assert!(linked.buffers.is_empty());
            match &first {
                None => first = Some(linked.kernel),
                Some(k_ptr) => assert!(Arc::ptr_eq(k_ptr, &linked.kernel), "circle {i} recompiled"),
            }
        }
    }

    #[test]
    fn a_uniform_and_a_constant_are_different_kernels() {
        let (k, _) = circle_of(true);
        let mut folded = ExprArena::new();
        let x = folded.push_var(0);
        let cx = folded.push_const(0.0);
        let r = folded.push_const(1.0);
        let cy = folded.push_const(0.0);
        let d = folded.push_binary(OpKind::Sub, x, cx);
        let scaled = folded.push_binary(OpKind::Mul, d, r);
        let froot = folded.push_binary(OpKind::Add, scaled, cy);
        let k_folded = Kernel::from_parts(folded, froot);
        assert!(!Arc::ptr_eq(&kernel_of(&k), &kernel_of(&k_folded)));
    }

    fn gather_over(decl: BufferDecl) -> Kernel {
        let mut a = ExprArena::new();
        let buf = a.declare_buffer(decl);
        let x = a.push_var(0);
        let y = a.push_var(1);
        let root = a.push_gather(buf, x, y);
        Kernel::from_parts(a, root)
    }

    #[test]
    fn bound_memory_of_one_shape_shares_code_and_of_another_does_not() {
        let decl = |width, height| BufferDecl {
            id: BufferIdentity::mint(),
            width,
            height,
        };
        let k1 = gather_over(decl(8, 4));
        let k2 = gather_over(decl(8, 4));
        let k3 = gather_over(decl(8, 5));
        let l1 = compile(&k1, TEST_SHAPE).expect("compile");
        let l2 = compile(&k2, TEST_SHAPE).expect("compile");
        let l3 = compile(&k3, TEST_SHAPE).expect("compile");
        assert!(
            Arc::ptr_eq(&l1.kernel, &l2.kernel),
            "two atlases of one shape are one kernel — the link tells them apart"
        );
        assert_ne!(l1.buffers[0].id, l2.buffers[0].id);
        assert!(
            !Arc::ptr_eq(&l1.kernel, &l3.kernel),
            "a buffer of different extents is different code"
        );
    }

    /// The link is by first occurrence in the canonical traversal, and the
    /// relinked arena's slots follow it regardless of declaration order.
    #[test]
    fn slots_follow_first_occurrence_not_declaration_order() {
        let (k, [cx, r, cy]) = circle_of(false);
        assert_eq!(k.uniforms(), &[r, cy, cx], "declared in the other order");
        let Canonical { uniforms, .. } = canon(&k);
        assert_eq!(uniforms, [cx, r, cy]);
        let (a, root) = k.parts();
        let (linked, lroot) = a.relink(root, &[], &uniforms);
        assert_eq!(linked.uniforms(), &[cx, r, cy]);
        let k_linked = Kernel::from_parts(linked, lroot);
        assert_eq!(
            canon(&k_linked).key,
            canon(&k).key,
            "relinking changes no structure"
        );
        // Both declaration orders canonicalize to one key — the shape, not
        // the table, is what the code is a function of.
        let (kb, _) = circle_of(true);
        assert_eq!(canon(&k).key, canon(&kb).key);
    }

    /// A table naming an instance the graph never reads — what `Kernel::at`
    /// leaves behind for every axis the receiver does not read — must link.
    /// The link omits it; the relink drops it with the garbage. Forced down
    /// the optimizer's bail path (`RawGather` is post-lowering, so the
    /// e-graph declines and the input arena, table and all, is what gets
    /// linked).
    #[test]
    fn a_declared_but_unread_uniform_links_on_the_bail_path() {
        let decl = |default| UniformDecl {
            id: UniformIdentity::mint(),
            default,
        };
        let (phase, scale) = (decl(0.0), decl(2.0));
        let mut a = ExprArena::new();
        let buf = a.declare_buffer(BufferDecl {
            id: BufferIdentity::mint(),
            width: 4,
            height: 1,
        });
        // Declared and read by a node nothing reaches — the `.at` shape.
        let dead = a.declare_uniform(phase);
        let _dead_read = a.push_uniform(dead);
        let live = a.declare_uniform(scale);
        let x = a.push_var(0);
        let b = a.push_buffer(buf);
        let g = a.push_binary(OpKind::RawGather, b, x);
        let s = a.push_uniform(live);
        let root = a.push_binary(OpKind::Mul, g, s);
        let k = Kernel::from_parts(a, root);
        assert_eq!(k.uniforms().len(), 2, "the table names both");

        let linked = compile(&k, TEST_SHAPE).expect("a dead declaration must not refuse to link");
        assert_eq!(
            linked.uniforms,
            [scale],
            "the link holds what the graph reads"
        );
        assert_eq!(linked.buffers.len(), 1);
    }
}
