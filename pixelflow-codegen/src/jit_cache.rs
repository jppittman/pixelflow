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
//! (docs/plans/2026-09-09-composition-is-linking.md §2). This cache is one
//! consumer of it.
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
use pixelflow_ir::arena::{BufferDecl, UniformDecl};
use pixelflow_ir::key::{Canonical, canonical};
use pixelflow_ir::{Environment, ExprBuilder, ExprData, ExprHandle, Node, Rooted};

static CACHE: OnceLock<Mutex<HashMap<Vec<u8>, Arc<CompiledKernel>>>> = OnceLock::new();

/// Rebuild a rooted graph with declarations in canonical first-occurrence
/// order.  The cache key deliberately ignores declaration identity, so a hit
/// must compile against dense slots rather than whichever order the first
/// caller happened to construct.  Construction stays inside `ExprBuilder`;
/// no caller observes or manipulates DAG storage.
fn canonical_graph(
    root: Node<'_, ExprData>,
    source: &Environment,
    buffers: &[BufferDecl],
    uniforms: &[UniformDecl],
) -> (Rooted<ExprData>, Environment) {
    let mut builder = ExprBuilder::new();
    let mut reachable = root.dag().side_table(false);
    for node in root.descendants() {
        reachable[node] = true;
    }
    let mut handles = root.dag().side_table(None::<ExprHandle>);
    let mut ordered: Vec<Node<'_, ExprData>> = root.descendants().collect();
    ordered.reverse();
    for node in ordered {
        if !reachable[node] {
            continue;
        }
        let handle = match *node {
            ExprData::Var(i) => builder.var(i),
            ExprData::Const(bits) => builder.constant(f32::from_bits(bits)),
            ExprData::Param(i) => builder.param(i),
            ExprData::Buffer(id) => {
                let decl = *source
                    .buffers
                    .get(id.0 as usize)
                    .expect("buffer slot in source environment");
                let canonical = buffers
                    .iter()
                    .find(|candidate| **candidate == decl)
                    .copied()
                    .expect("reachable buffer must be in canonical link table");
                builder.buffer(canonical)
            }
            ExprData::Uniform(id) => {
                let decl = *source
                    .uniforms
                    .get(id.0 as usize)
                    .expect("uniform slot in source environment");
                let canonical = uniforms
                    .iter()
                    .find(|candidate| **candidate == decl)
                    .copied()
                    .expect("reachable uniform must be in canonical link table");
                builder.uniform(canonical)
            }
            ExprData::Ref(key) => builder.reference(key),
            ExprData::Reduce(fold) => {
                let body = handles[node.children().next().expect("reduce body")]
                    .expect("reduce body copied before parent");
                builder.reduce(fold, body)
            }
            ExprData::Op(op) => {
                let children = node
                    .children()
                    .map(|child| handles[child].expect("child copied before parent"))
                    .collect::<Vec<_>>();
                match children.as_slice() {
                    [child] => builder.unary(op, *child),
                    [a, b] => builder.binary(op, *a, *b),
                    [a, b, c] => builder.ternary(op, *a, *b, *c),
                    many => builder.nary(op, many),
                }
            }
        };
        handles[node] = Some(handle);
    }
    let root = handles[root].expect("root copied into canonical graph");
    let graph = builder.finish_one(root);
    graph.into_parts()
}

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
/// Whatever [`emit::compile`] reports for the optimized rooted DAG.
///
/// # Panics
///
/// Panics if the rooted DAG — as handed in, or as saturation leaves it — names a
/// retired coordinate axis (`Var(2)`/`Var(3)`, the old Z and W). The
/// assertion itself lives one layer down, in
/// [`emit::compile`](crate::emit::compile), because that is the boundary
/// every route to machine code passes through and this is only one of them.
/// Compile a [`Kernel`](pixelflow_ir::Kernel) for a lattice of the given `shape`.
pub fn compile(kernel: &pixelflow_ir::Kernel, shape: LatticeShape) -> Result<Linked, CompileError> {
    let rooted = kernel.rooted().clone();
    let env = kernel.environment().clone();
    // References first, before the key or the link is read off anything. A
    // `Ref` is a leaf whose body — and whose buffer and uniform declarations
    // — are not in this DAG, so a key taken here would name a kernel other
    // than the one that gets emitted, and the link handed back would be
    // missing every slot the referent reads. Expanding *is* the linker
    // (docs/plans/2026-09-09-composition-is-linking.md §3), and it makes
    // "a reference is a kernel" true at this boundary rather than only in the
    // algebra: `Manifold::compile` needs to know nothing about it.
    //
    // Guarded rather than called unconditionally: the pass's own identity
    // path still clones the graph, and this runs on every compile including
    // the cache hits.
    let Canonical {
        mut key,
        buffers,
        uniforms,
    } = canonical(rooted.entry(), &env);

    // Optimize, link, then emit. This is not a step callers get to sequence:
    // a rooted graph reaching a backend unoptimized is never what anyone wanted,
    // and when the choice was on offer, two of the three production call
    // sites took the wrong one and compiled the terminal's cell-grid kernels
    // with no CSE and no FMA fusion. It is inside the compile entry because
    // that is the only place it cannot be forgotten.
    //
    // It bails to the graph as given for constructs the e-graph does not
    // model (a `Tuple` root; `Reduce` is unrolled ahead of saturation and
    // does optimize); those still compile, just without the extra fusion.
    //
    // The relink after it renumbers the tables into the canonical order the
    // key was built from — extraction redeclares identities in its own
    // walk order — without touching a node, so the bytes of a kernel that
    // declares neither a buffer nor a uniform are exactly what they were.
    let emit_fn = |rooted: Rooted<pixelflow_ir::ExprData>, env: pixelflow_ir::Environment| {
        let optimized = pixelflow_search::runtime::optimize_runtime_dag(&rooted, &env, shape);
        let (rooted, env) = optimized
            .as_deref()
            .map(|(rooted, env)| (rooted.clone(), env.clone()))
            .unwrap_or((rooted, env));
        let (rooted, env) = canonical_graph(rooted.entry(), &env, &buffers, &uniforms);
        emit::compile_dag(rooted.entry(), &env)
    };

    // Keyed on the rooted graph *as handed in*, before optimization, plus the shape.
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

    // Compile outside the lock so concurrent distinct-kernel constructions
    // don't serialize. A racing duplicate compile wastes work; the first
    // insertion wins so all callers share one region.
    let result = emit_fn(rooted, env)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::{ExprBuilder, Kernel, OpKind};

    const TEST_SHAPE: LatticeShape = LatticeShape::new([64, 64]);

    fn circle(garbage: bool) -> Kernel {
        let mut b = ExprBuilder::new();
        if garbage {
            let dead = b.constant(123.0);
            let _ = b.unary(OpKind::Sqrt, dead);
        }
        let x = b.var(0);
        let y = b.var(1);
        let x2 = b.binary(OpKind::Mul, x, x);
        let y2 = b.binary(OpKind::Mul, y, y);
        let sum = b.binary(OpKind::Add, x2, y2);
        let root = b.unary(OpKind::Sqrt, sum);
        Kernel::from_graph(b.finish_one(root))
    }

    #[test]
    fn canonical_shapes_share_compiled_code() {
        let first = compile(&circle(false), TEST_SHAPE).expect("compile").kernel;
        let second = compile(&circle(true), TEST_SHAPE).expect("compile").kernel;
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn distinct_shapes_do_not_share_compiled_code() {
        let first = compile(&circle(false), TEST_SHAPE).expect("compile").kernel;
        let mut b = ExprBuilder::new();
        let x = b.var(0);
        let y = b.var(1);
        let root = b.binary(OpKind::Sub, x, y);
        let second = compile(&Kernel::from_graph(b.finish_one(root)), TEST_SHAPE)
            .expect("compile")
            .kernel;
        assert!(!Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn lattice_extent_is_part_of_the_cache_key() {
        let kernel = circle(false);
        let first = compile(&kernel, TEST_SHAPE).expect("compile").kernel;
        let second = compile(&kernel, LatticeShape::new([65, 64]))
            .expect("compile")
            .kernel;
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(second.shape().extent(), [65, 64]);
    }

    #[test]
    fn cache_entries_are_monotonic_for_distinct_kernels() {
        let before = entry_count();
        let mut b = ExprBuilder::new();
        let x = b.var(0);
        let c = b.constant(424_242.0);
        let root = b.binary(OpKind::Mul, x, c);
        let _ = compile(&Kernel::from_graph(b.finish_one(root)), TEST_SHAPE).expect("compile");
        assert!(entry_count() > before);
    }

    #[test]
    fn canonical_link_reports_declared_slots() {
        let mut b = ExprBuilder::new();
        let x = b.var(0);
        let uniform = b.uniform(pixelflow_ir::arena::UniformDecl {
            id: pixelflow_ir::arena::UniformIdentity::mint(),
            default: 1.0,
        });
        let root = b.binary(OpKind::Mul, x, uniform);
        let kernel = Kernel::from_graph(b.finish_one(root));
        let linked = compile(&kernel, TEST_SHAPE).expect("compile");
        assert_eq!(linked.uniforms, kernel.uniforms());
        assert!(linked.buffers.is_empty());
    }
}
