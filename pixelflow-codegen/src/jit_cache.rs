//! Global JIT compile cache: identical kernels compile once.
//!
//! Skipping codegen entirely avoids recompiling a kernel that was already
//! compiled. The macro-emitted builders can hit this frequently: every
//! call of an N-param builder with the same arguments (window resizes), and
//! every structurally identical glyph kernel in a bake sweep, produces a
//! byte-identical schedule.
//!
//! Keys are the [`LatticeShape`] the kernel is compiled for — its extents,
//! so a lattice of a different size is a different kernel and a window
//! resize recompiles, by decision — plus the **canonical form of the
//! reachable subgraph**: nodes in ascending id order with ids remapped
//! dense. Construction garbage (dead
//! nodes left behind by `substitute_params` / splicing rebuilds) does not
//! perturb the key, so logically identical kernels hit regardless of build
//! history. Keys are compared by full equality — a hash collision can cause
//! a wasted probe, never wrong code.
//!
//! ## The link step
//!
//! A kernel that reads bound memory or a uniform names it by *identity*, and
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

use crate::CompiledKernel;
use crate::emit;
use crate::error::CompileError;
use pixelflow_ir::LatticeShape;
use pixelflow_ir::decl::{BufferDecl, UniformDecl};
use pixelflow_ir::{ExprData, Node, SideTable, Term};

static CACHE: OnceLock<Mutex<HashMap<Vec<u8>, Arc<CompiledKernel>>>> = OnceLock::new();

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

/// Compile the kernel rooted at `root` to an executable [`CompiledKernel`] (2D collapse loop)
/// for a lattice of the given `shape`, sharing previously compiled code for
/// canonically identical kernels at the same extents.
///
/// The returned `Arc` is the shared handle — two constructions of the same
/// kernel at the same shape yield pointer-equal manifolds — and the link
/// beside it says what this construction's slots bind.
///
/// # Errors
///
/// Whatever [`emit::compile`] reports for the (optimized) term.
///
/// # Panics
///
/// Panics if the term — as handed in, or as saturation leaves it — names a
/// retired coordinate axis (`Var(2)`/`Var(3)`, the old Z and W). The
/// assertion itself lives one layer down, in
/// [`emit::compile`](crate::emit::compile), because that is the boundary
/// every route to machine code passes through and this is only one of them.
/// Compile a [`Kernel`](pixelflow_ir::Kernel) for a lattice of the given `shape`.
pub fn compile(kernel: &pixelflow_ir::Kernel, shape: LatticeShape) -> Result<Linked, CompileError> {
    // One term, end to end: the key below and the code emitted at the bottom
    // are computed from this same value. They used to be two — a `Node`-based
    // canonical walk beside an `(&ExprArena, ExprId)` compile — and two
    // spellings of "the graph being compiled" is one chance for the key to
    // describe a different program than the bytes it names.
    let term = kernel.term();
    let Canonical {
        mut key,
        buffers,
        uniforms,
    } = canonical(term);

    // Keyed on the term *as handed in*, before optimization, plus the shape.
    // Optimization is a deterministic function of those two, so equal inputs
    // yield equal output and a hit skips the saturation as well as the codegen.
    key.extend_from_slice(&shape.key_bytes());
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache.lock().expect("jit_cache: lock poisoned").get(&key) {
        return Ok(Linked {
            kernel: hit.clone(),
            buffers,
            uniforms,
        });
    }

    // Optimize, link, then emit. This is not a step callers get to sequence:
    // a term reaching a backend unoptimized is never what anyone wanted,
    // and when the choice was on offer, two of the three production call
    // sites took the wrong one and compiled the terminal's cell-grid kernels
    // with no CSE and no FMA fusion. It is inside the compile entry because
    // that is the only place it cannot be forgotten.
    //
    // It bails to the term as given for constructs the e-graph does not
    // model (a `Tuple` root; `Reduce` is unrolled ahead of saturation and
    // does optimize); those still compile, just without the extra fusion.
    //
    // The relink after it renumbers the tables into the canonical order the
    // key was built from — extraction redeclares identities in its own
    // walk order — without touching a node, so the bytes of a kernel that
    // declares neither a buffer nor a uniform are exactly what they were.
    //
    // Compiled outside the lock so concurrent distinct-kernel constructions
    // don't serialize. A racing duplicate compile wastes work; the first
    // insertion wins so all callers share one region.
    let optimized = pixelflow_search::runtime::optimize_runtime_term(term, shape);
    let term = match optimized.as_deref() {
        Some((rooted, env)) => Term::new(rooted.entry(), env),
        None => term,
    };
    let result = if buffers.is_empty() && uniforms.is_empty() {
        emit::compile(term)?
    } else {
        let (relinked, env) = pixelflow_ir::relink(term, &buffers, &uniforms);
        emit::compile(Term::new(relinked.entry(), &env))?
    };
    let compiled = Arc::new(CompiledKernel::new(result.code, shape));
    let mut guard = cache.lock().expect("jit_cache: lock poisoned");
    let kernel = guard.entry(key).or_insert(compiled).clone();
    Ok(Linked {
        kernel,
        buffers,
        uniforms,
    })
}

/// Number of distinct kernels interned so far (test/telemetry hook).
#[must_use]
pub fn entry_count() -> usize {
    CACHE
        .get()
        .map(|c| c.lock().expect("jit_cache: lock poisoned").len())
        .unwrap_or(0)
}

/// The canonical form of a reachable subgraph: its key, and the buffer and
/// uniform declarations in the dense order the key numbers them by.
struct Canonical {
    key: Vec<u8>,
    buffers: Vec<BufferDecl>,
    uniforms: Vec<UniformDecl>,
}

/// Canonical serialization of the subgraph reachable from `term`'s root: nodes in
/// ascending topological order (in Dag, children always strictly precede parents),
/// child references remapped to dense indices, and buffer
/// and uniform leaves remapped to dense slots by first occurrence in that
/// same order — never by identity, which is what lets two compositions of
/// one shape share code.
fn canonical(term: Term<'_>) -> Canonical {
    let root = term.root();
    let dag = term.dag();
    let mut reachable = dag.side_table(false);
    for n in root.descendants() {
        reachable[n] = true;
    }

    // Dense remap in ascending id order (dag.iter() yields children strictly before parents).
    let mut dense = dag.side_table(u32::MAX);
    let mut next = 0u32;
    let mut key: Vec<u8> = Vec::with_capacity(dag.len() * 8);
    let mut buffers: Vec<BufferDecl> = Vec::new();
    let mut uniforms: Vec<UniformDecl> = Vec::new();

    let push_node = |key: &mut Vec<u8>, dense: &SideTable<u32>, n: Node<'_, ExprData>| {
        let d = dense[n];
        debug_assert_ne!(d, u32::MAX, "child densified before parent");
        key.extend_from_slice(&d.to_le_bytes());
    };

    /// The dense slot of `decl` in `table`, appending it on first sight.
    fn dense_slot<T: PartialEq + Copy>(table: &mut Vec<T>, decl: T) -> u16 {
        let slot = table.iter().position(|d| *d == decl).unwrap_or_else(|| {
            table.push(decl);
            table.len() - 1
        });
        u16::try_from(slot).expect("dense slot fits the table index width")
    }

    let env = term.env();

    for n in dag.iter() {
        if !reachable[n] {
            continue;
        }
        match *n.get() {
            ExprData::Var(i) => {
                key.push(0);
                key.push(i);
            }
            ExprData::Const(bits) => {
                key.push(1);
                key.extend_from_slice(&bits.to_le_bytes());
            }
            ExprData::Param(i) => {
                key.push(2);
                key.push(i);
            }
            ExprData::Op(op) => {
                let count = n.child_count();
                match count {
                    1 => key.push(3),
                    2 => key.push(4),
                    3 => key.push(5),
                    _ => {
                        key.push(6);
                        key.extend_from_slice(&op.marshal().to_bytes());
                        key.extend_from_slice(&(count as u32).to_le_bytes());
                        for child in n.children() {
                            push_node(&mut key, &dense, child);
                        }
                        dense[n] = next;
                        next += 1;
                        continue;
                    }
                }
                key.extend_from_slice(&op.marshal().to_bytes());
                for child in n.children() {
                    push_node(&mut key, &dense, child);
                }
            }
            // Slot by first occurrence, extents in the key: the code folds
            // its address arithmetic against them.
            ExprData::Buffer(b) => {
                let decl = env.buffer(b);
                key.push(7);
                key.extend_from_slice(&dense_slot(&mut buffers, decl).to_le_bytes());
                key.extend_from_slice(&decl.width.to_le_bytes());
                key.extend_from_slice(&decl.height.to_le_bytes());
            }
            // Offset by first occurrence; the default is the block's
            // business, not the code's.
            ExprData::Uniform(u) => {
                let decl = env.uniform(u);
                key.push(8);
                key.extend_from_slice(&dense_slot(&mut uniforms, decl).to_le_bytes());
            }
        }
        dense[n] = next;
        next += 1;
    }

    Canonical {
        key,
        buffers,
        uniforms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::Kernel;
    use pixelflow_ir::decl::{BufferIdentity, UniformIdentity};
    use pixelflow_ir::expr::{Environment, ExprBuilder};
    use pixelflow_ir::kind::OpKind;
    use pixelflow_ir::{Rooted, relink};

    const TEST_SHAPE: LatticeShape = LatticeShape::new([64, 64]);

    fn kernel_of(kernel: &Kernel) -> Arc<CompiledKernel> {
        compile(kernel, TEST_SHAPE).expect("compile").kernel
    }

    /// The term over a built `(Rooted, Environment)` pair.
    fn term_of(built: &(Rooted<ExprData>, Environment)) -> Term<'_> {
        Term::new(built.0.entry(), &built.1)
    }

    // The retired-axis backstop is exercised in `emit`'s own tests
    // (`compile_refuses_a_retired_coordinate_axis`): `Kernel::from_rooted`
    // refuses such a graph outright, so no `Kernel` naming one can be built
    // to hand to this module's entry point at all. The assertion still lives
    // in `emit::compile`, which is the boundary every route to machine code
    // passes through — including the ones that never build a `Kernel`.

    fn circle_kernel(garbage: bool) -> Kernel {
        let mut a = ExprBuilder::new();
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
        let (rooted, env) = a.finish(&[root]);
        Kernel::from_rooted(rooted, env)
    }

    #[test]
    fn identical_kernels_share_code() {
        let k1 = circle_kernel(false);
        let k2 = circle_kernel(true);
        let m1 = kernel_of(&k1);
        let m2 = kernel_of(&k2);
        assert!(
            Arc::ptr_eq(&m1, &m2),
            "canonically identical kernels must share one compiled region"
        );
    }

    #[test]
    fn distinct_kernels_do_not_collide() {
        let k1 = circle_kernel(false);
        let mut a2 = ExprBuilder::new();
        let x = a2.push_var(0);
        let y = a2.push_var(1);
        let r2 = a2.push_binary(OpKind::Sub, x, y);
        let (rooted, env) = a2.finish(&[r2]);
        let k2 = Kernel::from_rooted(rooted, env);
        let m1 = kernel_of(&k1);
        let m2 = kernel_of(&k2);
        assert!(!Arc::ptr_eq(&m1, &m2));
    }

    #[test]
    fn entry_count_is_monotonic_and_grows_on_new_kernels() {
        let mut a = ExprBuilder::new();
        let x = a.push_var(0);
        let k = a.push_const(424_242.0);
        let r = a.push_binary(OpKind::Mul, x, k);
        let (rooted, env) = a.finish(&[r]);
        let before = entry_count();
        let _m1 = kernel_of(&Kernel::from_rooted(rooted, env));
        let after_one = entry_count();
        assert!(
            after_one > before,
            "compiling a new kernel must grow the cache"
        );

        let mut a2 = ExprBuilder::new();
        let y = a2.push_var(1);
        let k2 = a2.push_const(535_353.0);
        let r2 = a2.push_binary(OpKind::Mul, y, k2);
        let (rooted2, env2) = a2.finish(&[r2]);
        let _m2 = kernel_of(&Kernel::from_rooted(rooted2, env2));
        let after_two = entry_count();
        assert!(
            after_two > after_one,
            "compiling another distinct kernel must grow the cache again"
        );
    }

    #[test]
    fn nary_reduce_children_affect_kernel_identity() {
        // Two back-to-back `Reduce` nodes in one graph: the second one's
        // children are a later span of the DAG's flat edge slab, so an
        // off-by-slicing bug in the second node's child range either indexes
        // out of bounds or silently drops its children from the key, making
        // two kernels that differ only in the second reduce collide.
        let mut a = ExprBuilder::new();
        let x = a.push_var(0);
        let body1 = a.push_binary(OpKind::Add, x, x);
        let r1 = a.push_reduce(OpKind::Add, 4, 3, body1);
        let body2 = a.push_binary(OpKind::Mul, x, x);
        let r2 = a.push_reduce(OpKind::Mul, 5, 6, body2);
        let root = a.push_binary(OpKind::Add, r1, r2);
        let (rooted, env) = a.finish(&[root]);

        let mut a2 = ExprBuilder::new();
        let x2 = a2.push_var(0);
        let body1b = a2.push_binary(OpKind::Add, x2, x2);
        let r1b = a2.push_reduce(OpKind::Add, 4, 3, body1b);
        let body2b = a2.push_binary(OpKind::Mul, x2, x2);
        let r2b = a2.push_reduce(OpKind::Mul, 5, 9, body2b); // extent differs only here
        let root2 = a2.push_binary(OpKind::Add, r1b, r2b);
        let (rooted2, env2) = a2.finish(&[root2]);

        let m1 = kernel_of(&Kernel::from_rooted(rooted, env));
        let m2 = kernel_of(&Kernel::from_rooted(rooted2, env2));
        assert!(
            !Arc::ptr_eq(&m1, &m2),
            "a change confined to the second reduce's extent must not share a cache entry"
        );
    }

    #[test]
    fn operand_order_flip_is_a_distinct_kernel() {
        let k1 = circle_kernel(false);

        let mut a3 = ExprBuilder::new();
        let x = a3.push_var(0);
        let y = a3.push_var(1);
        let x2 = a3.push_binary(OpKind::Mul, x, x);
        let y2 = a3.push_binary(OpKind::Mul, y, y);
        let s = a3.push_binary(OpKind::Add, y2, x2); // operand order flipped
        let r3 = a3.push_unary(OpKind::Sqrt, s);
        let (rooted, env) = a3.finish(&[r3]);

        let m1 = kernel_of(&k1);
        let m3 = kernel_of(&Kernel::from_rooted(rooted, env));
        assert!(
            !Arc::ptr_eq(&m1, &m3),
            "flipping operand order must not share a cache entry"
        );
    }

    #[test]
    fn same_kernel_at_two_extents_is_two_entries() {
        let k = circle_kernel(false);
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
        assert_eq!(frame.shape(), TEST_SHAPE);
        assert_eq!(wider.shape().extent(), [65, 64]);
    }

    // ─────────────────────── the link step ───────────────────────

    /// `(x − cx)·r + cy` over fresh uniform instances: one shape, many
    /// factors. `declared_first` flips the table order so the link, not the
    /// declaration order, is what the code is compiled against.
    fn circle_of(declared_first: bool) -> (Kernel, [UniformDecl; 3]) {
        let decl = |default| UniformDecl {
            id: UniformIdentity::mint(),
            default,
        };
        let (cx, cy, r) = (decl(0.0), decl(0.0), decl(1.0));
        let mut a = ExprBuilder::new();
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
        let (rooted, env) = a.finish(&[root]);
        (Kernel::from_rooted(rooted, env), [cx, r, cy])
    }

    /// Every instance shares the one region and gets its own link. The
    /// cache's *count* is asserted in `tests/one_compile_per_shape.rs`, a
    /// binary of its own: the count is process-global, and the other tests
    /// in this one compile kernels concurrently.
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
        let mut folded = ExprBuilder::new();
        let x = folded.push_var(0);
        let cx = folded.push_const(0.0);
        let r = folded.push_const(1.0);
        let cy = folded.push_const(0.0);
        let d = folded.push_binary(OpKind::Sub, x, cx);
        let scaled = folded.push_binary(OpKind::Mul, d, r);
        let froot = folded.push_binary(OpKind::Add, scaled, cy);
        let (rooted, env) = folded.finish(&[froot]);
        let k_folded = Kernel::from_rooted(rooted, env);
        assert!(!Arc::ptr_eq(&kernel_of(&k), &kernel_of(&k_folded)));
    }

    fn gather_over(decl: BufferDecl) -> Kernel {
        let mut a = ExprBuilder::new();
        let buf = a.declare_buffer(decl);
        let x = a.push_var(0);
        let y = a.push_var(1);
        let root = a.push_gather(buf, x, y);
        let (rooted, env) = a.finish(&[root]);
        Kernel::from_rooted(rooted, env)
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
    /// relinked term's slots follow it regardless of declaration order.
    #[test]
    fn slots_follow_first_occurrence_not_declaration_order() {
        let (k, [cx, r, cy]) = circle_of(false);
        assert_eq!(k.uniforms(), &[r, cy, cx], "declared in the other order");
        let Canonical { uniforms, .. } = canonical(k.term());
        assert_eq!(uniforms, [cx, r, cy]);
        let (relinked, lenv) = relink(k.term(), &[], &uniforms);
        assert_eq!(lenv.uniforms, [cx, r, cy]);
        assert_eq!(
            relinked.len(),
            k.dag().len(),
            "every node here is reachable, so relinking keeps them all"
        );
        let linked_pair = (relinked, lenv);
        assert_eq!(
            canonical(term_of(&linked_pair)).key,
            canonical(k.term()).key,
            "relinking changes no structure"
        );
        // Both declaration orders canonicalize to one key — the shape, not
        // the table, is what the code is a function of.
        let (kb, _) = circle_of(true);
        assert_eq!(canonical(k.term()).key, canonical(kb.term()).key);
    }

    /// A table naming an instance the graph never reads — what `Kernel::at`
    /// leaves behind for every axis the receiver does not read — must link.
    /// The link omits it; the relink drops it with the garbage. Forced down
    /// the optimizer's bail path (`RawGather` is post-lowering, so the
    /// e-graph declines and the input term, table and all, is what gets
    /// linked), which is where an unreachable declaration used to panic.
    #[test]
    fn a_declared_but_unread_uniform_links_on_the_bail_path() {
        let decl = |default| UniformDecl {
            id: UniformIdentity::mint(),
            default,
        };
        let (phase, scale) = (decl(0.0), decl(2.0));
        let mut a = ExprBuilder::new();
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
        let (rooted, env) = a.finish(&[root]);
        let k = Kernel::from_rooted(rooted, env);
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
