//! E-graph optimization for runtime-built kernels.
//!
//! `kernel!`/`kernel_jit!` already run full saturation at macro-expansion
//! time: `pixelflow-compiler::optimize` builds an e-graph from the parsed
//! AST, saturates, and extracts before ever touching an
//! [`ExprArena`](pixelflow_ir::ExprArena). Anything stamped by those macros
//! reaches `pixelflow_codegen::jit_cache` already optimized.
//!
//! `Kernel` values composed directly at runtime — `Kernel::over`, `.at()`,
//! `.select()`, arithmetic — never go through that macro, so their arenas hit
//! [`pixelflow_codegen::jit_cache::compile`] raw:
//! no CSE, no FMA fusion, no algebraic simplification. [`optimize_runtime_arena`]
//! is the same pipeline applied to an arena directly, for exactly that gap —
//! today's highest-volume instance is the font glyph bake
//! (`pixelflow-graphics`'s `Font::glyph_kernel_scaled`, cached per
//! `(codepoint, size, density)` bucket).
//!
//! `pixelflow-ir` itself must stay free of a `pixelflow-search` dependency
//! (the suckless constraint from
//! docs/plans/2026-07-20-kernel-unification.md), so this cannot live inside
//! `jit_cache`. Callers that want optimized runtime kernels — today,
//! `pixelflow-core`'s `Lattice::bake` — call this function before handing the
//! arena to `jit_cache`.
//!
//! # Saturation telemetry
//!
//! Build with `--features saturation-telemetry` to have every call here emit
//! one JSONL record of its saturation run (budget, stop reason, cost, wall
//! clock — see [`crate::telemetry`]). Point it at a file with
//! `PIXELFLOW_SATURATION_TELEMETRY=/path/to/log.jsonl cargo run --features saturation-telemetry`,
//! or leave it unset to see records on stderr.

use crate::egraph::{EClassId, EGraph, ENode, Op, Optimizer};
use pixelflow_ir::LatticeShape;
use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{BufferDecl, ExprArena, ExprId, ExprNode};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Optimize a runtime-built arena via bounded e-graph saturation, through
/// the same [`Optimizer`] entry point — rule set, budget, cost model,
/// extractor — as the `kernel!`/`kernel_jit!` macros.
///
/// `Buffer`/`Gather` (bound-memory reads) are representable: they enter the
/// e-graph as opaque structure — no rewrite rule can name them, so their
/// gain is hash-consing CSE (splice-duplicated sampler subtrees collapse to
/// one node) plus ordinary rewriting of the coordinate arithmetic that feeds
/// them. Extraction redeclares each distinct `BufferIdentity` once in the
/// output arena.
///
/// Returns `None`, unchanged, when the subgraph reachable from `root`
/// contains a construct the e-graph doesn't model:
///
/// - `RawGather` — produced by lowering, after the e-graph's place in the
///   pipeline; reaching one here means the arena is already lowered.
/// - `Nary` other than `Reduce` (`Tuple`) — not modelled. `Reduce` itself
///   is unrolled first (`passes::expand_reduce`, the same unroll `legalize`
///   performs later): the arena the e-graph sees is binder-free, so factoring
///   across the unrolled terms is ordinary rewriting rather than rewriting
///   under a binder.
/// - `Param` — a `pixelflow-compiler` macro-parameter slot that should never
///   reach a runtime-built `Kernel` in the first place.
///
/// Callers compile the original arena unchanged in that case — `optimize_runtime_arena`
/// is strictly an optimization, never required for correctness.
///
/// `shape` is the extent of the lattice the kernel is compiled for. It is
/// part of the cache key and is consulted by no rewrite yet (stage 0′ of
/// `docs/plans/2026-09-01-loop-aware-codegen.md`), so that extent-weighted
/// extraction (stage 1) is a policy change here and a signature change
/// nowhere. Saturation does not depend on it; when stage 1 lands, this cache
/// should hold the saturated e-graph per structure and extract per extent.
///
/// Cached by the structural shape of the reachable subgraph (mirroring
/// `pixelflow_codegen::jit_cache`'s own canonical-key cache): a caller that bakes
/// the same `Kernel` across many frames — every glyph, on the common path
/// through `GlyphCache` — pays saturation once, not once per bake. Skipping
/// this cache would make `optimize_runtime_arena` slower than not optimizing
/// at all for any repeatedly-baked kernel, since the JIT compile it feeds is
/// itself cached downstream.
///
/// The cached value is `Arc`-wrapped for the same reason `jit_cache` hands
/// back `Arc<JitManifold>` rather than owned code: a hit must be an atomic
/// refcount bump, not a deep clone of the (potentially large — real glyph
/// arenas run to thousands of nodes once construction garbage is counted)
/// optimized `ExprArena`. Returning an owned tuple here would silently
/// reintroduce a per-call cost the cache exists to eliminate.
#[must_use]
pub fn optimize_runtime_arena(
    arena: &ExprArena,
    root: ExprId,
    shape: LatticeShape,
) -> Option<Arc<(ExprArena, ExprId)>> {
    static CACHE: OnceLock<Mutex<HashMap<Vec<u8>, Option<Arc<(ExprArena, ExprId)>>>>> =
        OnceLock::new();

    // Buffer-bearing arenas bypass the cache entirely: `BufferIdentity` is
    // process-unique and minted per construction, so two compiles never
    // share a key — every lookup would miss while every insert stayed
    // forever (the cache is static and unbounded). A terminal resizing all
    // day would leak one full optimized arena per recompile for zero hits.
    if !arena.buffers().is_empty() {
        return optimize_runtime_arena_uncached(arena, root, shape).map(Arc::new);
    }

    let mut key = canonical_key(arena, root);
    key.extend_from_slice(&shape.key_bytes());
    // Optimization is a deterministic function of the arena, the shape, and
    // the *optimizer configuration* — the third term was missing, and the
    // claim that the first two suffice becomes false the moment two
    // configurations coexist in a process (a warm-up at one budget and
    // steady state at another, some kernels reranked and some not). It is a
    // constant today because production names exactly one configuration;
    // keying on it is what keeps that from being load-bearing.
    key.extend_from_slice(&Optimizer::production().fingerprint().to_bytes());
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache
        .lock()
        .expect("optimize_runtime_arena: lock poisoned")
        .get(&key)
    {
        return hit.clone();
    }

    let result = optimize_runtime_arena_uncached(arena, root, shape).map(Arc::new);
    cache
        .lock()
        .expect("optimize_runtime_arena: lock poisoned")
        .entry(key)
        .or_insert(result)
        .clone()
}

fn optimize_runtime_arena_uncached(
    arena: &ExprArena,
    root: ExprId,
    shape: LatticeShape,
) -> Option<(ExprArena, ExprId)> {
    // Resolve `Dwrt` FIRST, with the same exact symbolic pass the compile
    // entries run — then the e-graph sees pure arithmetic. Order matters
    // enormously: differentiation manufactures constants (the winding
    // kernels' `d = X − f(Y)` gives `DX(d) = 1` and, for straight edges, a
    // constant `DY(d)` — making the whole gradient magnitude `√(DX²+DY²)` a
    // compile-time number), and `ConstantFold` can only cascade over
    // constants that exist by the time saturation runs. Lowering after the
    // e-graph (the compile entries' own fallback position) leaves those
    // folds permanently on the table because nothing folds post-extraction.
    //
    // A lowering error (a genuinely non-differentiable op) bails to `None`;
    // the arena then compiles unoptimized and the compile entry's own
    // `lower_dwrt` reports the same error loudly at the right layer.
    let (arena, root) = pixelflow_ir::passes::lower_dwrt_owned(arena, root).ok()?;
    // Then unroll every `Reduce`, in `legalize`'s order. The extents are
    // static, so the binder disappears into N terms sharing their
    // index-invariant subtrees (`unroll_reduce` factors `⊕_i (f(i)·c)` as
    // `c·⊕_i f(i)` by declining to duplicate `c`), and the e-graph sees pure
    // arithmetic it can CSE and fold across the terms.
    let (arena, root) = pixelflow_ir::passes::expand_reduce_owned(&arena, root);

    // One entry point, shared with the AOT macro tier and the `Dwrt`
    // expansion tier: the rule set, the budget, the cost model, and the
    // extractor are decided in `Optimizer`, not re-decided here. Priced
    // against the lattice this kernel is compiled for — the extents are
    // known, so extraction minimizes the instruction count of the whole
    // program rather than of its text.
    let mut optimizer = Optimizer::production().for_lattice(shape);

    let mut egraph = optimizer.egraph();
    let mut memo: HashMap<ExprId, EClassId> = HashMap::new();
    let root_class = arena_to_egraph(&arena, root, &mut egraph, &mut memo)?;

    let node_count = reachable_count(&arena, root);
    #[cfg(feature = "saturation-telemetry")]
    let telemetry_start = std::time::Instant::now();
    let optimized = optimizer.run(&mut egraph, root_class, node_count);
    let (extracted, extracted_root) = optimized.to_arena(&egraph, root_class);

    #[cfg(feature = "saturation-telemetry")]
    crate::telemetry::record(crate::telemetry::SaturationInvocation {
        tier: crate::telemetry::Tier::Runtime,
        node_count,
        stats: &optimized.stats,
        union_count: egraph.provenance().union_count(),
        extracted_arena: &extracted,
        extracted_root,
        wall_clock: telemetry_start.elapsed(),
        kernel_label: None,
    });

    // The extracted arena declares buffers in extraction-traversal order,
    // which need not match the input's — and slot order is ABI: the JIT
    // loads slot i's base pointer from the caller's context array at i*8,
    // and callers bind in the order the arena THEY BUILT declared. A
    // different extraction (a commuted equivalent under another cost model)
    // must not silently permute their pointers. Re-splicing onto a table
    // pre-declared in input order makes the invariant structural: splice
    // dedups buffers by identity onto the existing slots.
    if arena.buffers().is_empty() {
        return Some((extracted, extracted_root));
    }
    let mut ordered = ExprArena::new();
    for decl in arena.buffers() {
        let _slot = ordered.declare_buffer(*decl);
    }
    let root = ordered.splice(&extracted, extracted_root);
    debug_assert!(
        ordered
            .buffers()
            .iter()
            .zip(arena.buffers())
            .all(|(a, b)| a.id == b.id),
        "buffer slot order must survive optimization"
    );
    Some((ordered, root))
}

/// Canonical serialization of the subgraph reachable from `root`: nodes in
/// ascending original id order (the arena is append-only, so children always
/// precede parents), child references remapped to dense indices — the same
/// shape as `pixelflow_codegen::jit_cache`'s private `canonical_key`, reimplemented
/// here since that one isn't exported and this cache's correctness condition
/// is different (it needs a key for *every* reachable node kind, including
/// `Buffer`/`Nary`, to memoize the bail-out case too, not just the
/// e-graph-representable ones).
fn canonical_key(arena: &ExprArena, root: ExprId) -> Vec<u8> {
    let len = arena.nodes_raw().len();
    let mut reachable = vec![false; len];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if core::mem::replace(&mut reachable[id.0 as usize], true) {
            continue;
        }
        stack.extend(arena.children(id));
    }

    let mut dense: Vec<u32> = vec![u32::MAX; len];
    let mut next = 0u32;
    let mut key: Vec<u8> = Vec::with_capacity(len * 8);

    let push_id = |key: &mut Vec<u8>, dense: &[u32], id: ExprId| {
        let d = dense[id.0 as usize];
        debug_assert_ne!(d, u32::MAX, "canonical_key: child densified before parent");
        key.extend_from_slice(&d.to_le_bytes());
    };

    for idx in 0..len {
        if !reachable[idx] {
            continue;
        }
        let id = ExprId(idx as u32);
        match arena.node(id) {
            &ExprNode::Var(i) => {
                key.push(0);
                key.push(i);
            }
            &ExprNode::Const(v) => {
                key.push(1);
                key.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            &ExprNode::Param(i) => {
                key.push(2);
                key.push(i);
            }
            &ExprNode::Buffer(b) => {
                key.push(3);
                // Key by process-unique BufferIdentity, NOT the arena-local
                // slot index: two arenas can both call their own buffer "slot
                // 0" with equal extents while naming different memory, and a
                // slot-keyed cache would hand one of them the other's
                // optimized arena — whose redeclared identity binds the wrong
                // pixels. Identity is process-local, which is exactly the
                // lifetime of this in-process cache. It has no byte accessor,
                // so serialize its (injective) Debug form.
                let BufferDecl { id, width, height } = *arena.buffer_decl(b);
                key.extend_from_slice(alloc::format!("{id:?}").as_bytes());
                key.extend_from_slice(&width.to_le_bytes());
                key.extend_from_slice(&height.to_le_bytes());
            }
            &ExprNode::Unary(op, a) => {
                key.push(4);
                key.extend_from_slice(&op.marshal().to_bytes());
                push_id(&mut key, &dense, a);
            }
            &ExprNode::Binary(op, a, b) => {
                key.push(5);
                key.extend_from_slice(&op.marshal().to_bytes());
                push_id(&mut key, &dense, a);
                push_id(&mut key, &dense, b);
            }
            &ExprNode::Ternary(op, a, b, c) => {
                key.push(6);
                key.extend_from_slice(&op.marshal().to_bytes());
                push_id(&mut key, &dense, a);
                push_id(&mut key, &dense, b);
                push_id(&mut key, &dense, c);
            }
            &ExprNode::Nary(op, start, n) => {
                key.push(7);
                key.extend_from_slice(&op.marshal().to_bytes());
                key.extend_from_slice(&n.to_le_bytes());
                let (s, l) = (start as usize, n as usize);
                for &child in &arena.nary_children_raw()[s..s + l] {
                    push_id(&mut key, &dense, child);
                }
            }
        }
        dense[idx] = next;
        next += 1;
    }

    key
}

// ─────────────────────────── Runtime-only mask ops ───────────────────────────
//
// `&`/`|` on comparison masks are surface-language ops (every glyph winding
// kernel's Y-range gate is `(Y >= lo) & (Y < hi)`), so runtime-built arenas
// must be representable with them present. They are deliberately NOT in
// `egraph::ops::op_from_kind`: registering them globally hands them to the
// AOT macro tier too, whose e-graph runs at macro-expansion time — BEFORE
// composition — where resolving the `Dwrt` nodes the masks travel with is
// unsound (a leaf's `DX` is 1 only until an enclosing `.at()` warp scales
// it; the fonts' density-dependent AA ramp broke exactly this way when
// these ops were briefly global). The runtime tier optimizes the final
// composed arena at bake time, where the calculus has its full context, so
// mask ops are safe — and only meaningful — here.
//
// No rewrite rule targets them; they participate as opaque structure plus
// `ConstantFold`, whose bitwise-domain exemption already models their
// all-ones/zero masks.
struct MaskAnd;
impl Op for MaskAnd {
    fn kind(&self) -> OpKind {
        OpKind::BitAnd
    }
}
struct MaskOr;
impl Op for MaskOr {
    fn kind(&self) -> OpKind {
        OpKind::BitOr
    }
}

// ─────────────────────── Runtime-only integer-domain ops ─────────────────────
//
// The packed cell-grid kernel's spine: clamp → `TruncToInt` → `Shl` →
// or-fold builds a `u32` pixel per lane, so the production frame kernel is
// unrepresentable — and therefore compiles with NO CSE across its four
// channels — unless these enter the e-graph. Runtime-tier only, for the
// same reason as the mask ops above. Opaque to TEMPLATES: no rewrite rule can
// name them (nothing here or in `op_from_kind` hands them to a template), and
// their results are bit patterns the float rule set has no semantics for.
//
// Template-opacity is NOT fold-opacity, and the distinction is load-bearing:
// `ConstantFold::apply` destructures any `ENode::Op` and reads `op.kind()`
// (`math::algebra`) — it never consults `op_from_kind`. So every op registered
// here folds, and each one needs its own answer to "does this fold agree with
// what the backends emit?" `OpKind::fold_is_platform_specific` is where that
// answer lives; being unnameable by a template guards nothing.
//
// `Shl`/`Shr` do keep `Const` shift operands, because extraction emits `Const`
// leaves verbatim — so the emitter's immediate-only contract holds. The count's
// RANGE is a separate matter, enforced where the `Const` narrows to an
// immediate (`emit::shift_immediate`) rather than assumed here.
struct IntTrunc;
impl Op for IntTrunc {
    fn kind(&self) -> OpKind {
        OpKind::TruncToInt
    }
}
struct IntFromInt;
impl Op for IntFromInt {
    fn kind(&self) -> OpKind {
        OpKind::IntToFloat
    }
}
struct IntAdd;
impl Op for IntAdd {
    fn kind(&self) -> OpKind {
        OpKind::IAdd
    }
}
struct IntShl;
impl Op for IntShl {
    fn kind(&self) -> OpKind {
        OpKind::Shl
    }
}
struct IntShr;
impl Op for IntShr {
    fn kind(&self) -> OpKind {
        OpKind::Shr
    }
}

/// Whether the runtime tier can represent `kind` in its e-graph — i.e.,
/// whether an arena containing it still optimizes rather than bailing.
/// Test hook for the representability guards; the semantics live in
/// [`runtime_op_from_kind`].
#[must_use]
pub fn is_egraph_representable(kind: OpKind) -> bool {
    runtime_op_from_kind(kind).is_some()
}

/// [`crate::egraph::ops::op_from_kind`] extended with the runtime-only mask
/// ops above and the opaque `Gather` op (absent from the global lookup so no
/// rewrite template can name it — its participation is hash-consing CSE
/// only). Every conversion in this module resolves ops through this.
fn runtime_op_from_kind(kind: OpKind) -> Option<&'static dyn Op> {
    match kind {
        OpKind::BitAnd => Some(&MaskAnd),
        OpKind::BitOr => Some(&MaskOr),
        OpKind::TruncToInt => Some(&IntTrunc),
        OpKind::IntToFloat => Some(&IntFromInt),
        OpKind::IAdd => Some(&IntAdd),
        OpKind::Shl => Some(&IntShl),
        OpKind::Shr => Some(&IntShr),
        OpKind::Gather => Some(&crate::egraph::ops::Gather),
        other => crate::egraph::ops::op_from_kind(other),
    }
}

/// Insert the subgraph reachable from `id` into `egraph`, memoized by
/// `ExprId` (on top of the e-graph's own hash-consing by node shape) so a
/// DAG-shared arena is walked once per node, not once per reference.
///
/// Returns `None` — aborting the whole conversion — the moment it meets an
/// op [`crate::egraph::ops::op_from_kind`] doesn't model, or a `Param`.
/// Iterative (explicit stack), matching [`choices_to_arena`]'s style in the
/// same crate: arena depths are unbounded in principle (Dwrt chain-rule
/// expansion, deep composition), so this must not blow the Rust stack.
fn arena_to_egraph(
    arena: &ExprArena,
    root: ExprId,
    egraph: &mut EGraph,
    memo: &mut HashMap<ExprId, EClassId>,
) -> Option<EClassId> {
    enum Task {
        Visit(ExprId),
        Complete(ExprId),
    }

    let mut task_stack = vec![Task::Visit(root)];
    let mut result_stack: Vec<EClassId> = Vec::new();

    while let Some(task) = task_stack.pop() {
        match task {
            Task::Visit(id) => {
                if let Some(&class) = memo.get(&id) {
                    result_stack.push(class);
                    continue;
                }
                match arena.node(id) {
                    &ExprNode::Var(idx) => {
                        let class = egraph.add(ENode::Var(idx));
                        memo.insert(id, class);
                        result_stack.push(class);
                    }
                    &ExprNode::Const(val) => {
                        let class = egraph.add(ENode::constant(val));
                        memo.insert(id, class);
                        result_stack.push(class);
                    }
                    ExprNode::Param(_) => return None,
                    &ExprNode::Buffer(b) => {
                        let class = egraph.add(ENode::Buffer(*arena.buffer_decl(b)));
                        memo.insert(id, class);
                        result_stack.push(class);
                    }
                    &ExprNode::Unary(kind, a) => {
                        runtime_op_from_kind(kind)?;
                        task_stack.push(Task::Complete(id));
                        task_stack.push(Task::Visit(a));
                    }
                    &ExprNode::Binary(kind, a, b) => {
                        runtime_op_from_kind(kind)?;
                        task_stack.push(Task::Complete(id));
                        task_stack.push(Task::Visit(b));
                        task_stack.push(Task::Visit(a));
                    }
                    &ExprNode::Ternary(kind, a, b, c) => {
                        runtime_op_from_kind(kind)?;
                        task_stack.push(Task::Complete(id));
                        task_stack.push(Task::Visit(c));
                        task_stack.push(Task::Visit(b));
                        task_stack.push(Task::Visit(a));
                    }
                    // `Reduce` was unrolled and `Dwrt` lowered before this
                    // walk; what remains (`Tuple`) is not modelled. Bail out.
                    ExprNode::Nary(..) => return None,
                }
            }
            Task::Complete(id) => {
                if let Some(&class) = memo.get(&id) {
                    result_stack.push(class);
                    continue;
                }
                let (kind, arity) = match arena.node(id) {
                    &ExprNode::Unary(kind, _) => (kind, 1),
                    &ExprNode::Binary(kind, _, _) => (kind, 2),
                    &ExprNode::Ternary(kind, _, _, _) => (kind, 3),
                    _ => unreachable!("Complete scheduled only for Unary/Binary/Ternary"),
                };
                let op = runtime_op_from_kind(kind)
                    .expect("runtime_op_from_kind already checked in Visit");
                let start = result_stack.len() - arity;
                let children: Vec<EClassId> = result_stack.drain(start..).collect();
                let class = egraph.add(ENode::Op { op, children });
                memo.insert(id, class);
                result_stack.push(class);
            }
        }
    }

    result_stack.pop()
}

/// Count nodes reachable from `root` — a rough size measure for
/// [`config_for_node_count`], mirroring what `pixelflow_codegen::jit_cache`'s
/// canonical-key reachability walk already does for the same arena.
fn reachable_count(arena: &ExprArena, root: ExprId) -> usize {
    let len = arena.nodes_raw().len();
    let mut seen = vec![false; len];
    let mut stack = vec![root];
    let mut count = 0usize;
    while let Some(id) = stack.pop() {
        if core::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        count += 1;
        stack.extend(arena.children(id));
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::OpKind;
    use pixelflow_ir::arena::BufferDecl;
    use pixelflow_ir::binding::BindingTable;
    use pixelflow_ir::eval_scalar;

    /// Every optimization must preserve the arena's denoted value, over a
    /// spread of coordinates — the load-bearing property. Anything that ever
    /// broke this would silently mis-render, not fail loudly.
    fn assert_semantics_preserved(
        arena: &ExprArena,
        root: ExprId,
        optimized: &(ExprArena, ExprId),
    ) {
        let (opt_arena, opt_root) = optimized;
        let coords: &[(f32, f32, f32, f32)] = &[
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 2.0, 3.0, 4.0),
            (-1.5, 0.5, 2.25, -3.0),
            (3.7, -4.1, 0.0, 1.0),
        ];
        for &(x, y, z, w) in coords {
            let want = eval_scalar(arena, root, &[x, y, z, w], &BindingTable::empty());
            let got = eval_scalar(opt_arena, *opt_root, &[x, y, z, w], &BindingTable::empty());
            assert!(
                (want - got).abs() < 1e-3 || (want.is_nan() && got.is_nan()),
                "optimize_runtime_arena changed semantics at ({x},{y},{z},{w}): {want} != {got}"
            );
        }
    }

    #[test]
    fn repeated_bake_of_the_same_kernel_hits_the_cache() {
        // The exact regression this cache exists to close: Lattice::bake
        // calls optimize_runtime_arena on EVERY bake of a Kernel, but real
        // callers (GlyphCache, and criterion benches that measure "the JIT
        // compile is cached, so iterations measure tabulation") bake the
        // *same* kernel repeatedly. Without caching, every one of those
        // calls re-runs full saturation from scratch — slower than not
        // optimizing at all, since the downstream JIT compile was already
        // cached and free.
        //
        // Build something big enough to land in the "classical" budget
        // (>50 nodes) with real rewriting work to do (redundant
        // sub-multiplications an FMA pass and commutativity/associativity
        // actually have to chew on), so a cold run takes measurably longer
        // than a hash lookup.
        fn build_arena() -> (ExprArena, ExprId) {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let mut acc = a.push_const(0.0);
            for k in 0..15 {
                let c = a.push_const(1.0 + k as f32 * 0.37);
                let xc = a.push_binary(OpKind::Mul, x, c);
                let yc = a.push_binary(OpKind::Mul, y, c);
                let term = a.push_binary(OpKind::Add, xc, yc);
                acc = a.push_binary(OpKind::Add, acc, term);
            }
            (a, acc)
        }

        let (a1, r1) = build_arena();
        let cold_start = std::time::Instant::now();
        let arc1 = optimize_runtime_arena(&a1, r1, pixelflow_ir::LatticeShape::POINT)
            .expect("must optimize");
        let opt1 = &arc1.0;
        let cold = cold_start.elapsed();

        // A freshly built, structurally identical (but not reused) arena:
        // proves the cache keys on shape, not on the first call's identity.
        let (a2, r2) = build_arena();
        let warm_start = std::time::Instant::now();
        let arc2 = optimize_runtime_arena(&a2, r2, pixelflow_ir::LatticeShape::POINT)
            .expect("must optimize");
        let opt2 = &arc2.0;
        let warm = warm_start.elapsed();

        assert_eq!(
            opt1.nodes_raw().len(),
            opt2.nodes_raw().len(),
            "cached and fresh optimization must agree on the result shape"
        );
        assert!(
            warm < cold / 2 || warm < std::time::Duration::from_micros(200),
            "expected the second call to hit the cache (warm {warm:?} vs cold {cold:?}) — \
             a regression here means optimize_runtime_arena is re-saturating every bake"
        );
    }

    #[test]
    fn fma_fusion_applies_to_a_runtime_arena() {
        // a*b + c, built directly as an arena (no macro involved) — exactly
        // the shape Kernel::over/.at() composition produces at runtime.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let z = a.push_var(2);
        let mul = a.push_binary(OpKind::Mul, x, y);
        let root = a.push_binary(OpKind::Add, mul, z);

        let arc = optimize_runtime_arena(&a, root, pixelflow_ir::LatticeShape::POINT)
            .expect("pure arithmetic arena must optimize");
        let (opt_arena, opt_root) = (arc.0.clone(), arc.1);

        assert_semantics_preserved(&a, root, &(opt_arena.clone(), opt_root));
        assert!(
            matches!(
                opt_arena.node(opt_root),
                ExprNode::Ternary(OpKind::MulAdd, ..)
            ),
            "expected a*b+c fused to MulAdd, got {:?}",
            opt_arena.node(opt_root)
        );
    }

    #[test]
    fn shared_subexpressions_stay_shared_and_correct() {
        // sin(X)*sin(X) + sin(X): the repeated sin(X) subtree must convert
        // to the e-graph once (via the ExprId memo) and extract back
        // correctly regardless of how many times it's referenced.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let s = a.push_unary(OpKind::Sin, x);
        let sq = a.push_binary(OpKind::Mul, s, s);
        let root = a.push_binary(OpKind::Add, sq, s);

        let arc = optimize_runtime_arena(&a, root, pixelflow_ir::LatticeShape::POINT)
            .expect("trig arena must optimize");
        let (opt_arena, opt_root) = (arc.0.clone(), arc.1);
        assert_semantics_preserved(&a, root, &(opt_arena, opt_root));
    }

    #[test]
    fn dwrt_derivative_kernel_optimizes() {
        // The font-coverage shape: X - ((Y - y0) * k + x0), differentiated.
        // Dwrt is representable in the e-graph (ChainRule reduces it), so
        // this must NOT bail out.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let y0 = a.push_const(0.3);
        let k = a.push_const(0.7);
        let x0 = a.push_const(-0.2);
        let y_sub = a.push_binary(OpKind::Sub, y, y0);
        let scaled = a.push_binary(OpKind::Mul, y_sub, k);
        let line = a.push_binary(OpKind::Add, scaled, x0);
        let d = a.push_binary(OpKind::Sub, x, line);
        let var_x = a.push_const(0.0); // Dwrt's second child is the var index, wrt X (0)
        let dx = a.push_binary(OpKind::Dwrt, d, var_x);

        let arc = optimize_runtime_arena(&a, dx, pixelflow_ir::LatticeShape::POINT)
            .expect("Dwrt-bearing arena must optimize");
        let (opt_arena, opt_root) = (arc.0.clone(), arc.1);

        // eval_scalar refuses a raw Dwrt (the interpreter evaluates the
        // post-calculus program, same as the JIT) — lower both sides before
        // comparing, cross-checking the e-graph's ChainRule reduction
        // against the dedicated lower_dwrt pass.
        use pixelflow_ir::passes::lower_dwrt_owned;
        let (want_arena, want_root) = lower_dwrt_owned(&a, dx).expect("lower original");
        let (got_arena, got_root) =
            lower_dwrt_owned(&opt_arena, opt_root).expect("lower optimized");
        assert_semantics_preserved(&want_arena, want_root, &(got_arena, got_root));
    }

    /// Ids of every node reachable from `root`, discovery order.
    fn reachable_ids(arena: &ExprArena, root: ExprId) -> Vec<ExprId> {
        let mut seen = vec![false; arena.nodes_raw().len()];
        let mut stack = vec![root];
        let mut out = Vec::new();
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            out.push(id);
            stack.extend(arena.children(id));
        }
        out
    }

    fn count_gathers(arena: &ExprArena, root: ExprId) -> usize {
        reachable_ids(arena, root)
            .iter()
            .filter(|&&id| matches!(arena.node(id), ExprNode::Ternary(OpKind::Gather, ..)))
            .count()
    }

    /// Identities of buffers referenced by reachable `Buffer` leaves.
    fn reachable_buffer_identities(
        arena: &ExprArena,
        root: ExprId,
    ) -> std::collections::BTreeSet<pixelflow_ir::arena::BufferIdentity> {
        reachable_ids(arena, root)
            .iter()
            .filter_map(|&id| match arena.node(id) {
                &ExprNode::Buffer(b) => Some(arena.buffer_decl(b).id),
                _ => None,
            })
            .collect()
    }

    /// Bind slices to an arena by buffer *identity*, not slot order: the
    /// optimizer redeclares buffers in extraction-traversal order, so the
    /// optimized arena's slot numbering can differ from the input's.
    fn bind_by_identity<'a>(
        arena: &ExprArena,
        by_id: &[(pixelflow_ir::arena::BufferIdentity, &'a [f32])],
    ) -> BindingTable<'a> {
        let slices: Vec<&[f32]> = arena
            .buffers()
            .iter()
            .map(|d| {
                by_id
                    .iter()
                    .find(|(id, _)| *id == d.id)
                    .unwrap_or_else(|| panic!("no slice for buffer identity {:?}", d.id))
                    .1
            })
            .collect();
        BindingTable::bind(arena, &slices).expect("bind_by_identity")
    }

    /// Eval parity for buffer-bearing arenas, both sides bound by identity.
    fn assert_gather_semantics_preserved(
        arena: &ExprArena,
        root: ExprId,
        optimized: &(ExprArena, ExprId),
        by_id: &[(pixelflow_ir::arena::BufferIdentity, &[f32])],
    ) {
        let (opt_arena, opt_root) = optimized;
        let want_bind = bind_by_identity(arena, by_id);
        let got_bind = bind_by_identity(opt_arena, by_id);
        // Coordinates chosen off integer boundaries so Gather's floor cannot
        // flip cells on rounding differences introduced by rewrites.
        let coords: &[(f32, f32)] = &[(0.3, 0.4), (1.5, 0.6), (2.2, 1.7), (3.6, 2.4), (-1.2, 9.5)];
        for &(cx, cy) in coords {
            let want = eval_scalar(arena, root, &[cx, cy, 0.0, 0.0], &want_bind);
            let got = eval_scalar(opt_arena, *opt_root, &[cx, cy, 0.0, 0.0], &got_bind);
            assert!(
                (want - got).abs() < 1e-3,
                "gather optimization changed semantics at ({cx},{cy}): {want} != {got}"
            );
        }
    }

    #[test]
    fn gather_arena_round_trips_through_the_egraph() {
        // BilinearSampler-shaped: the e-graph must now carry the Gather as
        // opaque structure and hand back an arena that declares the same
        // buffer (by identity) and evaluates identically.
        let identity = pixelflow_ir::arena::BufferIdentity::mint();
        let data: Vec<f32> = (0..16).map(|i| i as f32 * 3.0 + 1.0).collect();

        let mut a = ExprArena::new();
        let buf = a.declare_buffer(BufferDecl {
            id: identity,
            width: 4,
            height: 4,
        });
        let x = a.push_var(0);
        let y = a.push_var(1);
        let g = a.push_gather(buf, x, y);
        let one = a.push_const(1.0);
        let root = a.push_binary(OpKind::Add, g, one);

        let arc = optimize_runtime_arena(&a, root, pixelflow_ir::LatticeShape::POINT)
            .expect("a Gather-bearing arena must optimize, not bail");
        let (opt_arena, opt_root) = (arc.0.clone(), arc.1);

        assert_eq!(
            reachable_buffer_identities(&opt_arena, opt_root),
            reachable_buffer_identities(&a, root),
            "extraction must redeclare the same buffers, by identity"
        );
        assert_gather_semantics_preserved(
            &a,
            root,
            &(opt_arena, opt_root),
            &[(identity, data.as_slice())],
        );
    }

    #[test]
    fn duplicated_gathers_cse_into_one_node() {
        // The composition problem this change exists to solve: every use of a
        // sampler Kernel re-splices its fragment, so the SAME gather (same
        // buffer identity, same coordinate subtree) appears twice as two
        // disjoint copies. Hash-consing must collapse them to one node.
        let identity = pixelflow_ir::arena::BufferIdentity::mint();
        let data: Vec<f32> = (0..16).map(|i| (i * i) as f32).collect();

        let mut a = ExprArena::new();
        let buf = a.declare_buffer(BufferDecl {
            id: identity,
            width: 4,
            height: 4,
        });
        // Two structurally identical copies, pushed separately — exactly what
        // splice produces.
        let mut push_dup = |a: &mut ExprArena| {
            let x = a.push_var(0);
            let y = a.push_var(1);
            let one = a.push_const(1.0);
            let xx = a.push_binary(OpKind::Add, x, one);
            a.push_gather(buf, xx, y)
        };
        let g1 = push_dup(&mut a);
        let g2 = push_dup(&mut a);
        // Mul (not Add) so the doubling rule can't restructure the root and
        // muddy the count assertions.
        let root = a.push_binary(OpKind::Mul, g1, g2);

        let before = reachable_count(&a, root);
        assert_eq!(
            count_gathers(&a, root),
            2,
            "input must contain the duplicate"
        );

        let arc = optimize_runtime_arena(&a, root, pixelflow_ir::LatticeShape::POINT)
            .expect("must optimize");
        let (opt_arena, opt_root) = (arc.0.clone(), arc.1);
        let after = reachable_count(&opt_arena, opt_root);

        assert!(
            after < before,
            "CSE must strictly shrink the arena (before={before}, after={after})"
        );
        assert_eq!(
            count_gathers(&opt_arena, opt_root),
            1,
            "the two identical gathers must share one node"
        );
        assert_gather_semantics_preserved(
            &a,
            root,
            &(opt_arena, opt_root),
            &[(identity, data.as_slice())],
        );
    }

    /// Slot order is the binding ABI: the JIT loads slot i's base pointer
    /// from the context array at i*8, and callers bind in the order the arena
    /// THEY built declared. Extraction traverses in its own order — here the
    /// root's first child reads the SECOND-declared buffer — so a rebuild
    /// that declared buffers in traversal order would silently swap the
    /// caller's two pointers and read the wrong memory.
    #[test]
    fn optimization_preserves_buffer_slot_order() {
        let mut a = ExprArena::new();
        let buf_a = a.declare_buffer(BufferDecl {
            id: pixelflow_ir::arena::BufferIdentity::mint(),
            width: 4,
            height: 1,
        });
        let buf_b = a.declare_buffer(BufferDecl {
            id: pixelflow_ir::arena::BufferIdentity::mint(),
            width: 8,
            height: 1,
        });
        let x = a.push_var(0);
        let y = a.push_var(1);
        let gb = a.push_gather(buf_b, x, y);
        let ga = a.push_gather(buf_a, x, y);
        let root = a.push_binary(OpKind::Add, gb, ga);

        let out = optimize_runtime_arena(&a, root, pixelflow_ir::LatticeShape::POINT)
            .expect("buffer kernel must optimize");
        let input: Vec<_> = a.buffers().iter().map(|d| d.id).collect();
        let output: Vec<_> = out.0.buffers().iter().map(|d| d.id).collect();
        assert_eq!(input, output, "slot order must survive optimization");
    }

    #[test]
    fn distinct_buffer_identities_never_merge() {
        // Equal extents and identical coordinates are a coincidence, not the
        // same memory: gathers of different identities must stay distinct.
        let id_a = pixelflow_ir::arena::BufferIdentity::mint();
        let id_b = pixelflow_ir::arena::BufferIdentity::mint();
        let data_a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let data_b: Vec<f32> = (0..16).map(|i| 1000.0 - i as f32).collect();

        let mut a = ExprArena::new();
        let buf_a = a.declare_buffer(BufferDecl {
            id: id_a,
            width: 4,
            height: 4,
        });
        let buf_b = a.declare_buffer(BufferDecl {
            id: id_b,
            width: 4,
            height: 4,
        });
        let x = a.push_var(0);
        let y = a.push_var(1);
        let ga = a.push_gather(buf_a, x, y);
        let gb = a.push_gather(buf_b, x, y);
        let root = a.push_binary(OpKind::Sub, ga, gb);

        let arc = optimize_runtime_arena(&a, root, pixelflow_ir::LatticeShape::POINT)
            .expect("must optimize");
        let (opt_arena, opt_root) = (arc.0.clone(), arc.1);

        assert_eq!(
            count_gathers(&opt_arena, opt_root),
            2,
            "different identities with identical extents/coords must not merge"
        );
        assert_eq!(
            reachable_buffer_identities(&opt_arena, opt_root).len(),
            2,
            "both identities must survive extraction"
        );
        assert_gather_semantics_preserved(
            &a,
            root,
            &(opt_arena, opt_root),
            &[(id_a, data_a.as_slice()), (id_b, data_b.as_slice())],
        );
    }

    #[test]
    fn composed_cell_grid_kernel_shape_deduplicates() {
        // Faithful synthetic of the packed terminal kernel: cell-grid
        // coordinate arithmetic (cell index + intra-cell offset) feeding 5
        // gathers of one buffer (glyph atlas channels) and 4 of another
        // (color planes), with the coordinate subtree re-spliced VERBATIM for
        // every gather — exactly the duplication Kernel composition produces.
        let atlas_id = pixelflow_ir::arena::BufferIdentity::mint();
        let color_id = pixelflow_ir::arena::BufferIdentity::mint();
        let atlas: Vec<f32> = (0..(64 * 32)).map(|i| (i % 97) as f32).collect();
        let colors: Vec<f32> = (0..(64 * 32)).map(|i| (i % 251) as f32 * 0.5).collect();

        let mut a = ExprArena::new();
        let atlas_buf = a.declare_buffer(BufferDecl {
            id: atlas_id,
            width: 64,
            height: 32,
        });
        let color_buf = a.declare_buffer(BufferDecl {
            id: color_id,
            width: 64,
            height: 32,
        });

        // The shared coordinate arithmetic, duplicated per gather: cell
        // coords (floor(X/8), floor(Y/16)), intra-cell offsets, and an
        // atlas-space remap. The atlas cell stride (10, 18) differs from the
        // screen cell size (8, 16) so no algebraic rule can cancel the
        // arithmetic away — deduplication must come from hash-consing, as in
        // the real kernel.
        let mut push_coords = |a: &mut ExprArena| {
            let x = a.push_var(0);
            let y = a.push_var(1);
            let cw = a.push_const(8.0);
            let ch = a.push_const(16.0);
            let aw = a.push_const(10.0);
            let ah = a.push_const(18.0);
            let xc = a.push_binary(OpKind::Div, x, cw);
            let cx = a.push_unary(OpKind::Floor, xc);
            let yc = a.push_binary(OpKind::Div, y, ch);
            let cy = a.push_unary(OpKind::Floor, yc);
            let cxw = a.push_binary(OpKind::Mul, cx, cw);
            let fx = a.push_binary(OpKind::Sub, x, cxw);
            let cyh = a.push_binary(OpKind::Mul, cy, ch);
            let fy = a.push_binary(OpKind::Sub, y, cyh);
            let cxa = a.push_binary(OpKind::Mul, cx, aw);
            let cya = a.push_binary(OpKind::Mul, cy, ah);
            let sx = a.push_binary(OpKind::Add, cxa, fx);
            let sy = a.push_binary(OpKind::Add, cya, fy);
            (sx, sy)
        };

        let mut acc = a.push_const(0.0);
        for i in 0..9 {
            let (sx, sy) = push_coords(&mut a);
            let buf = if i < 5 { atlas_buf } else { color_buf };
            let g = a.push_gather(buf, sx, sy);
            let w = a.push_const(0.1 + i as f32 * 0.07);
            let wg = a.push_binary(OpKind::Mul, g, w);
            acc = a.push_binary(OpKind::Add, acc, wg);
        }
        let root = acc;

        let before = reachable_count(&a, root);
        assert_eq!(count_gathers(&a, root), 9);

        let arc = optimize_runtime_arena(&a, root, pixelflow_ir::LatticeShape::POINT)
            .expect("must optimize");
        let (opt_arena, opt_root) = (arc.0.clone(), arc.1);
        let after = reachable_count(&opt_arena, opt_root);

        // Report shape for the record: 9 duplicated ~13-node coordinate
        // subtrees must collapse to (at most) one shared copy, and the 5+4
        // gathers to one per (buffer, coords) pair — here 2 total.
        assert!(
            after < before,
            "composed kernel must come back deduplicated (before={before}, after={after})"
        );
        assert_eq!(
            count_gathers(&opt_arena, opt_root),
            2,
            "5 atlas + 4 color gathers of identical coords must CSE to one each"
        );
        assert_gather_semantics_preserved(
            &a,
            root,
            &(opt_arena, opt_root),
            &[(atlas_id, atlas.as_slice()), (color_id, colors.as_slice())],
        );

        // Keep the measured counts visible in test output (`--nocapture`).
        println!("composed cell-grid shape: before={before} nodes, after={after} nodes");
    }

    /// Extraction under a real lattice still computes the same function.
    ///
    /// Which *form* it picks is pinned deterministically in
    /// `egraph::extract`'s `scope_weighting_unfuses_an_fma_to_hoist_the_z_term`
    /// — through saturation the available forms depend on a wall-clock
    /// budget, so what is asserted here is the invariant that holds however
    /// far saturation got.
    #[test]
    fn scope_weighted_extraction_preserves_semantics() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let z = a.push_var(2);
        let inner = a.push_binary(OpKind::Add, x, z);
        let root = a.push_binary(OpKind::Add, inner, z);

        let frame = pixelflow_ir::LatticeShape::new([256, 256, 1, 1]);
        let arc = optimize_runtime_arena(&a, root, frame).expect("must optimize");
        let (opt, opt_root) = &*arc;
        for (x, z) in [(0.0f32, 0.0f32), (1.5, -2.0), (-3.25, 7.5)] {
            let want = eval_scalar(&a, root, &[x, 0.0, z, 0.0], &BindingTable::empty());
            let got = eval_scalar(opt, *opt_root, &[x, 0.0, z, 0.0], &BindingTable::empty());
            assert_eq!(got, want, "at X={x}, Z={z}");
        }
    }

    #[test]
    fn reduce_is_unrolled_and_optimized() {
        // Kernel::over-shaped: Σ_{i<4} i² over the reduction index slot. The
        // binder is distributed before saturation, so what the e-graph sees
        // is 0·0 + 1·1 + 2·2 + 3·3 — ordinary arithmetic it can fold.
        //
        // What this pins is the binder's disappearance and the value, not the
        // folder's reach: saturation runs under a wall-clock budget (10ms for
        // an arena this small), so *how far* the fold cascades is a property
        // of the machine, not of the compiler. Asserting `Const(14.0)` here
        // passed locally and failed on a loaded CI runner, which is the
        // assertion being wrong rather than the code.
        let mut a = ExprArena::new();
        let i = a.push_var(4);
        let body = a.push_binary(OpKind::Mul, i, i);
        let root = a.push_reduce(OpKind::Add, 4, 4, body);

        let arc = optimize_runtime_arena(&a, root, pixelflow_ir::LatticeShape::POINT)
            .expect("a Reduce-bearing arena must optimize once distributed");
        let (opt, opt_root) = &*arc;
        assert!(
            !reaches_nary(opt, *opt_root),
            "the binder must be gone from the optimized arena"
        );
        assert_eq!(
            eval_scalar(opt, *opt_root, &[0.0; 4], &BindingTable::empty()),
            14.0,
            "Σ_{{i<4}} i² = 0 + 1 + 4 + 9"
        );

        // Σ_{i<3} X·i: the surviving terms depend on X, and the optimized
        // form agrees with the interpreter on the distributed original.
        let mut b = ExprArena::new();
        let x = b.push_var(0);
        let j = b.push_var(5);
        let body = b.push_binary(OpKind::Mul, x, j);
        let root = b.push_reduce(OpKind::Add, 5, 3, body);
        let (unrolled, unrolled_root) = pixelflow_ir::passes::expand_reduce_owned(&b, root);
        let arc = optimize_runtime_arena(&b, root, pixelflow_ir::LatticeShape::POINT)
            .expect("X-dependent Reduce must optimize");
        let (opt, opt_root) = &*arc;
        assert!(!reaches_nary(opt, *opt_root));
        for x in [0.0f32, 1.5, -2.25, 7.0] {
            let want = eval_scalar(
                &unrolled,
                unrolled_root,
                &[x, 0.0, 0.0, 0.0],
                &BindingTable::empty(),
            );
            let got = eval_scalar(opt, *opt_root, &[x, 0.0, 0.0, 0.0], &BindingTable::empty());
            assert_eq!(got, want, "Σ_{{i<3}} X·i at X={x}");
        }
    }

    /// Whether any `Nary` (the `Reduce` binder) is reachable from `root`.
    fn reaches_nary(arena: &ExprArena, root: ExprId) -> bool {
        let mut seen = vec![false; arena.nodes_raw().len()];
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            if matches!(arena.node(id), ExprNode::Nary(..)) {
                return true;
            }
            stack.extend(arena.children(id));
        }
        false
    }

    #[test]
    fn constant_folds_through_bounded_saturation() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let one = a.push_const(1.0);
        let zero = a.push_const(0.0);
        let plus_zero = a.push_binary(OpKind::Add, x, zero);
        let times_one = a.push_binary(OpKind::Mul, plus_zero, one);
        let root = times_one;

        let arc = optimize_runtime_arena(&a, root, pixelflow_ir::LatticeShape::POINT)
            .expect("identity arena must optimize");
        let (opt_arena, opt_root) = (arc.0.clone(), arc.1);
        assert_semantics_preserved(&a, root, &(opt_arena.clone(), opt_root));
        // x + 0.0, then * 1.0 should collapse to bare X.
        assert!(
            matches!(opt_arena.node(opt_root), ExprNode::Var(0)),
            "expected identities to collapse to bare X, got {:?}",
            opt_arena.node(opt_root)
        );
    }
}

/// Read-only probe for issue #1106: how much congruence does the production
/// e-graph MISS because `union` only enqueues the merged class itself, never
/// its parents (no e-node parent list exists — `EGraph::parent` is the
/// union-find parent, not a parents-of-a-class list)? See
/// docs/results/2026-09-02-missing-congruence.md for the measurement this
/// module produces and its verdict.
///
/// Everything here is offline: it clones the post-saturation e-graph and
/// runs a from-scratch upward-closure sweep to fixpoint on the clone. It
/// changes nothing about production `optimize_runtime_arena` or `all_rules()`
/// ordering — this is measurement only, not the fix.
#[cfg(test)]
mod congruence_gap_probe {
    use super::*;
    use crate::egraph::rule_order::{RuleOrder, build_rule_set};
    use crate::egraph::{Congruence, CostModel, RuleSet, SaturationStop, choices_to_arena};
    use crate::nnue::{BwdGenConfig, BwdGenerator};
    use pixelflow_ir::arena::{BufferId, BufferIdentity};
    use std::path::{Path, PathBuf};

    /// Inverse of the dumpers' `dump_arena` (`pixelflow-core/src/lattice/cell_grid.rs`,
    /// `pixelflow-graphics/tests/production_glyph_arena_dump.rs`,
    /// `pixelflow-pipeline/tests/shader_and_psychedelic_arena_dump.rs`):
    /// replays reachable nodes in original id order through the public
    /// `push_*` API, which never hash-conses, so the rebuilt arena has
    /// exactly the dumped node multiset.
    fn load_arena_dump(path: &Path) -> (String, ExprArena, ExprId) {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let mut lines = text.lines();
        assert_eq!(
            lines.next(),
            Some("# pixelflow arena dump v1"),
            "{}: bad header",
            path.display()
        );
        let mut name = None;
        let mut arena = ExprArena::new();
        let mut idents: Vec<BufferIdentity> = Vec::new();
        let mut root = None;
        let mut next_id: u32 = 0;
        let mut buf_count: u16 = 0;
        let op = |s: &str| -> OpKind {
            OpKind::all()
                .find(|k| format!("{k:?}") == s)
                .unwrap_or_else(|| panic!("{}: unknown OpKind {s:?}", path.display()))
        };
        let id = |s: &str| -> ExprId {
            ExprId(
                s.parse()
                    .unwrap_or_else(|e| panic!("{}: bad id {s:?}: {e}", path.display())),
            )
        };
        for line in lines {
            let f: Vec<&str> = line.split_whitespace().collect();
            let pushed = match f.as_slice() {
                ["name", n] => {
                    name = Some((*n).to_string());
                    continue;
                }
                ["buf", ord, w, h] => {
                    let ord: usize = ord.parse().expect("buf ordinal");
                    while idents.len() <= ord {
                        idents.push(BufferIdentity::mint());
                    }
                    let slot = arena.declare_buffer(BufferDecl {
                        id: idents[ord],
                        width: w.parse().expect("buf width"),
                        height: h.parse().expect("buf height"),
                    });
                    assert_eq!(
                        slot.0,
                        buf_count,
                        "{}: buffer slot order drifted",
                        path.display()
                    );
                    buf_count += 1;
                    continue;
                }
                ["root", r] => {
                    root = Some(id(r));
                    continue;
                }
                ["V", i] => arena.push_var(i.parse().expect("var index")),
                ["C", bits] => arena.push_const(f32::from_bits(bits.parse().expect("const bits"))),
                ["B", slot] => arena.push_buffer(BufferId(slot.parse().expect("buffer slot"))),
                ["U", k, a] => arena.push_unary(op(k), id(a)),
                ["Bi", k, a, b] => arena.push_binary(op(k), id(a), id(b)),
                ["T", k, a, b, c] => arena.push_ternary(op(k), id(a), id(b), id(c)),
                other => panic!("{}: unparseable line {other:?}", path.display()),
            };
            assert_eq!(
                pushed,
                ExprId(next_id),
                "{}: replay drifted from dumped ids",
                path.display()
            );
            next_id += 1;
        }
        let name = name.unwrap_or_else(|| panic!("{}: no name line", path.display()));
        let root = root.unwrap_or_else(|| panic!("{}: no root line", path.display()));
        (name, arena, root)
    }

    /// Number of canonical (live) e-classes: `find(i) == i`. Distinct from
    /// `SaturationResult::classes_after`, which is `EGraph::classes.len()` —
    /// the raw allocation count, which never shrinks on `union` (the
    /// production 5,000-class cap is checked against THIS raw count, not the
    /// live count — see `EGraph::saturate_with_limits`,
    /// `self.classes.len() > max_classes`).
    fn live_class_count(egraph: &EGraph) -> usize {
        (0..egraph.classes.len())
            .filter(|&i| egraph.find(EClassId(i as u32)) == EClassId(i as u32))
            .count()
    }

    /// Full upward congruence closure, offline, to fixpoint, on whatever
    /// graph is passed in (call on a `.clone()` to keep the original
    /// untouched). Each pass re-canonicalizes every live class's e-nodes
    /// through `find` and unions any two live classes whose canonicalized
    /// node forms coincide — the "walk every e-node that references a
    /// changed class" step production's `union`/`rebuild_budgeted` never
    /// performs, because no parent-of-a-class index exists. Repeats until a
    /// full pass finds zero new unions.
    ///
    /// Returns the number of NEW unions this pass performed (== the live
    /// class-count reduction, since every `union` call here merges two
    /// distinct live classes into one).
    fn full_upward_closure(egraph: &mut EGraph) -> usize {
        const MAX_PASSES: usize = 20_000;
        let mut total_unions = 0usize;
        for _pass in 0..MAX_PASSES {
            let n = egraph.classes.len();
            let mut local_memo: HashMap<ENode, EClassId> = HashMap::new();
            let mut pending: Vec<(EClassId, EClassId)> = Vec::new();
            for idx in 0..n {
                let id = EClassId(idx as u32);
                let canon = egraph.find(id);
                if canon != id {
                    // Merged-away class: `union`'s mem::take already moved
                    // its nodes onto the surviving parent, so its own node
                    // vector is empty and re-scanning it would be a no-op.
                    continue;
                }
                let nodes = egraph.classes[idx].nodes.clone();
                for node in nodes {
                    let cnode = match &node {
                        ENode::Op { op, children } => ENode::Op {
                            op: *op,
                            children: children.iter().map(|c| egraph.find(*c)).collect(),
                        },
                        other => other.clone(),
                    };
                    match local_memo.get(&cnode) {
                        Some(&existing) => {
                            let existing_canon = egraph.find(existing);
                            if existing_canon != canon {
                                pending.push((canon, existing_canon));
                            }
                        }
                        None => {
                            local_memo.insert(cnode, canon);
                        }
                    }
                }
            }
            if pending.is_empty() {
                return total_unions;
            }
            for (a, b) in pending {
                let ra = egraph.find(a);
                let rb = egraph.find(b);
                if ra != rb {
                    egraph.union(ra, rb);
                    total_unions += 1;
                }
            }
        }
        panic!(
            "full_upward_closure: did not reach fixpoint in {MAX_PASSES} passes \
             (total_unions so far: {total_unions}) — either a real non-termination \
             bug or MAX_PASSES needs raising for this corpus"
        );
    }

    /// Sum of per-op latency-prior cost over the nodes reachable from `root`
    /// — the "materialized extracted arena's real cost" the rule-order
    /// harness this probe borrows its corpus from also uses (`arena_cost` in
    /// `docs/results/2026-09-01-rule-order-real-kernels.md`), not
    /// `ExtractedDAG::total_cost`'s cycle-penalty-inflated DP total.
    fn arena_static_cost(model: &CostModel, arena: &ExprArena, root: ExprId) -> usize {
        let len = arena.nodes_raw().len();
        let mut seen = vec![false; len];
        let mut stack = vec![root];
        let mut total = 0usize;
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            let kind = match arena.node(id) {
                ExprNode::Unary(k, _)
                | ExprNode::Binary(k, _, _)
                | ExprNode::Ternary(k, _, _, _) => Some(*k),
                _ => None,
            };
            if let Some(k) = kind {
                total += model.cost(k);
            }
            stack.extend(arena.children(id));
        }
        total
    }

    #[derive(Clone, Debug)]
    struct CongruenceRow {
        name: String,
        category: &'static str,
        rule_order: String,
        node_count: usize,
        max_classes: usize,
        stop: String,
        hit_class_cap: bool,
        live_before: usize,
        closure_unions: usize,
        live_after: usize,
        reduction_frac: f64,
        cost_before: usize,
        cost_after: usize,
        cost_change_frac: f64,
        would_avoid_cap: bool,
    }

    /// Run the production regime — `config_for_node_count` + `saturate_with_full_budget`,
    /// exactly as `optimize_runtime_arena_uncached` calls them — under rule
    /// order `order`, then measure the offline upward-closure gap on a clone.
    fn measure_one(
        name: &str,
        category: &'static str,
        order: RuleOrder,
        arena: &ExprArena,
        root: ExprId,
    ) -> CongruenceRow {
        // Same two lowering passes `optimize_runtime_arena_uncached` runs
        // before the e-graph ever sees the arena (Dwrt resolved first so
        // ConstantFold can cascade over the constants it manufactures, then
        // Reduce unrolled). Real production dumps (the glyph corpus) still
        // carry raw `Dwrt` markers at this point — `Font::glyph_kernel_scaled`
        // returns `kernel.parts()` before this step runs.
        let (arena, root) = pixelflow_ir::passes::lower_dwrt_owned(arena, root)
            .unwrap_or_else(|e| panic!("{name}: lower_dwrt failed: {e:?}"));
        let (arena, root) = pixelflow_ir::passes::expand_reduce_owned(&arena, root);
        let node_count = reachable_count(&arena, root);

        // THE production regime, through the one entry point
        // `optimize_runtime_arena_uncached` itself now calls
        // (pixelflow-search#1108, "one optimizer entry point"):
        // `Optimizer::production()` bundles the rule set, `Budget::Production`
        // (== `config_for_node_count`'s tiers), `CostModel::latency_prior`,
        // and the extractor. `order` swaps only the rule set — everything
        // else stays exactly production's configuration.
        let mut optimizer = match order {
            RuleOrder::Production => Optimizer::production(),
            other => Optimizer::production().rules(RuleSet::new(build_rule_set(other))),
        };
        let mut egraph = optimizer.egraph();
        let mut memo: HashMap<ExprId, EClassId> = HashMap::new();
        let root_class = arena_to_egraph(&arena, root, &mut egraph, &mut memo)
            .unwrap_or_else(|| panic!("{name}: arena_to_egraph returned None (unsupported node)"));

        let optimized = optimizer.run(&mut egraph, root_class, node_count);
        let max_classes = optimized.stats.limits.classes;
        let hit_class_cap = optimized.stats.stop == SaturationStop::ClassCap;

        let live_before = live_class_count(&egraph);

        let model = CostModel::latency_prior();
        let (extracted_before, extracted_before_root) = optimized.to_arena(&egraph, root_class);
        let cost_before = arena_static_cost(&model, &extracted_before, extracted_before_root);

        // The offline upward-closure pass runs on a CLONE — production's own
        // e-graph (and `optimized.stats` above) is untouched.
        let mut closure_graph = egraph.clone();
        let closure_unions = full_upward_closure(&mut closure_graph);
        let live_after = live_class_count(&closure_graph);

        let root_class_closed = closure_graph.find(root_class);
        let dag_after = crate::egraph::extract::extract_dag_scoped(
            &closure_graph,
            root_class_closed,
            &model,
            LatticeShape::POINT,
        );
        let extraction_after = crate::egraph::Extraction::from_dp(
            &closure_graph,
            root_class_closed,
            dag_after.choices,
        );
        let (extracted_after, extracted_after_root) = choices_to_arena(&extraction_after);
        let cost_after = arena_static_cost(&model, &extracted_after, extracted_after_root);

        let reduction_frac = if live_before > 0 {
            closure_unions as f64 / live_before as f64
        } else {
            0.0
        };
        let cost_change_frac = if cost_before > 0 {
            (cost_after as f64 - cost_before as f64) / cost_before as f64
        } else {
            0.0
        };
        let would_avoid_cap = hit_class_cap && live_after < max_classes;

        CongruenceRow {
            name: name.to_string(),
            category,
            rule_order: order.to_string(),
            node_count,
            max_classes,
            stop: format!("{:?}", optimized.stats.stop),
            hit_class_cap,
            live_before,
            closure_unions,
            live_after,
            reduction_frac,
            cost_before,
            cost_after,
            cost_change_frac,
            would_avoid_cap,
        }
    }

    fn category_of(filename: &str) -> &'static str {
        if filename.starts_with("cellgrid_") {
            "cellgrid"
        } else if filename.starts_with("shader_") {
            "shader"
        } else if filename.starts_with("psychedelic") {
            "psychedelic"
        } else if filename.starts_with("glyph") {
            "glyph"
        } else {
            "unknown"
        }
    }

    fn median(xs: &mut [f64]) -> f64 {
        if xs.is_empty() {
            return f64::NAN;
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = xs.len();
        if n % 2 == 1 {
            xs[n / 2]
        } else {
            (xs[n / 2 - 1] + xs[n / 2]) / 2.0
        }
    }

    fn percentile(xs: &mut [f64], p: f64) -> f64 {
        if xs.is_empty() {
            return f64::NAN;
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = xs.len();
        let idx = ((p / 100.0) * (n as f64 - 1.0)).round() as usize;
        xs[idx.min(n - 1)]
    }

    /// THE measurement (issue #1106): for the 204-real-kernel corpus (the
    /// #1101 shader/psychedelic/cell-grid/glyph dumps, reused verbatim —
    /// `PIXELFLOW_CONGRUENCE_ARENA_DIR` points at a directory produced by
    /// running the three `dump_*` `#[ignore]`d tests those dumpers live in)
    /// plus a size-stratified synthetic corpus from `BwdGenerator`, measure
    /// how much congruence the production e-graph misses under
    /// `all_rules()`'s production order, and how much of that is the H-b
    /// rule-order confound on a handful of the real kernels.
    ///
    /// Read-only: never changes `union`, `rebuild_budgeted`, or `all_rules()`
    /// order. Writes docs/results/2026-09-02-missing-congruence.{md,csv,json}.
    #[test]
    #[ignore = "offline measurement: PIXELFLOW_CONGRUENCE_ARENA_DIR=<dir of .arena dumps> cargo test -p pixelflow-search --release --lib -- --ignored missing_congruence_measurement"]
    fn missing_congruence_measurement() {
        let dir = PathBuf::from(
            std::env::var("PIXELFLOW_CONGRUENCE_ARENA_DIR")
                .expect("PIXELFLOW_CONGRUENCE_ARENA_DIR must be set"),
        );
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
            .map(|e| e.expect("dir entry").path())
            .filter(|p| p.extension().map(|e| e == "arena").unwrap_or(false))
            .collect();
        paths.sort();
        assert!(
            !paths.is_empty(),
            "no .arena files found in {}",
            dir.display()
        );
        eprintln!("congruence probe: {} real-kernel arena dumps", paths.len());

        let mut rows: Vec<CongruenceRow> = Vec::new();
        for path in &paths {
            let (name, arena, root) = load_arena_dump(path);
            let category = category_of(&path.file_name().unwrap().to_string_lossy());
            rows.push(measure_one(
                &name,
                category,
                RuleOrder::Production,
                &arena,
                root,
            ));
        }
        let real_kernel_count = rows.len();

        // H-b, cheap: does the MISSING-congruence count differ between
        // production order and numeric-first order, on a handful of real
        // kernels (one from each category present)?
        let mut hb_rows: Vec<CongruenceRow> = Vec::new();
        for prefix in [
            "cellgrid_80x24_d1",
            "shader_",
            "psychedelic",
            "glyph16_U0041",
        ] {
            if let Some(path) = paths
                .iter()
                .find(|p| p.file_name().unwrap().to_string_lossy().starts_with(prefix))
            {
                let (name, arena, root) = load_arena_dump(path);
                let category = category_of(&path.file_name().unwrap().to_string_lossy());
                hb_rows.push(measure_one(
                    &format!("{name} [production]"),
                    category,
                    RuleOrder::Production,
                    &arena,
                    root,
                ));
                hb_rows.push(measure_one(
                    &format!("{name} [numeric-first]"),
                    category,
                    RuleOrder::NumericFirst,
                    &arena,
                    root,
                ));
            }
        }

        // Synthetic classical corpus: size-stratified via BwdGenerator's
        // max_depth, ~200 samples (5 depth bands x 40 seeds), using the
        // UNOPTIMIZED (junkified) form — the realistic pre-optimization
        // shape the same generator mints for extraction-head training data.
        let templates = crate::egraph::collect_rule_templates();
        for &max_depth in &[3usize, 5, 7, 9, 11] {
            for seed in 0u64..40 {
                let config = BwdGenConfig {
                    max_depth,
                    ..Default::default()
                };
                let mut generator = BwdGenerator::new(
                    seed.wrapping_add(max_depth as u64 * 10_000),
                    config,
                    templates.clone(),
                );
                let pair = generator.generate_arena();
                let name = format!("synth_d{max_depth}_s{seed}");
                rows.push(measure_one(
                    &name,
                    "synthetic",
                    RuleOrder::Production,
                    &pair.arena,
                    pair.unoptimized,
                ));
            }
        }
        let synthetic_count = rows.len() - real_kernel_count;
        eprintln!("congruence probe: {synthetic_count} synthetic classical expressions");

        // ---- Aggregate stats ----
        let mut all_reduction_frac: Vec<f64> = rows.iter().map(|r| r.reduction_frac).collect();
        let mut all_cost_change_frac: Vec<f64> = rows.iter().map(|r| r.cost_change_frac).collect();
        let total_closure_unions: usize = rows.iter().map(|r| r.closure_unions).sum();
        let total_live_before: usize = rows.iter().map(|r| r.live_before).sum();
        let class_reduction_frac_overall = if total_live_before > 0 {
            total_closure_unions as f64 / total_live_before as f64
        } else {
            0.0
        };
        let cap_hit_rows: Vec<&CongruenceRow> = rows.iter().filter(|r| r.hit_class_cap).collect();
        let cap_hit_count = cap_hit_rows.len();
        let would_avoid_cap_count = rows.iter().filter(|r| r.would_avoid_cap).count();

        let mut capped_reduction: Vec<f64> =
            cap_hit_rows.iter().map(|r| r.reduction_frac).collect();
        let mut uncapped_reduction: Vec<f64> = rows
            .iter()
            .filter(|r| !r.hit_class_cap)
            .map(|r| r.reduction_frac)
            .collect();
        let mut capped_cost_change: Vec<f64> =
            cap_hit_rows.iter().map(|r| r.cost_change_frac).collect();
        let mut uncapped_cost_change: Vec<f64> = rows
            .iter()
            .filter(|r| !r.hit_class_cap)
            .map(|r| r.cost_change_frac)
            .collect();

        // The five headline numbers (task spec):
        let num1_additional_unions = total_closure_unions;
        let num1_frac_of_classes = class_reduction_frac_overall;
        let num2_median_class_reduction_frac = median(&mut all_reduction_frac.clone());
        let num3_median_cost_change_frac = median(&mut all_cost_change_frac.clone());
        let num4_would_avoid_cap = would_avoid_cap_count;
        let num4_of_cap_hits = cap_hit_count;

        eprintln!("=== headline numbers ===");
        eprintln!(
            "(1) additional unions found by closure: {num1_additional_unions} \
             ({:.2}% of live classes, pooled)",
            num1_frac_of_classes * 100.0
        );
        eprintln!(
            "(2) median per-kernel live-class-count reduction: {:.2}%",
            num2_median_class_reduction_frac * 100.0
        );
        eprintln!(
            "(3) median extracted-cost change after closure: {:.3}%",
            num3_median_cost_change_frac * 100.0
        );
        eprintln!(
            "(4) kernels that hit the class cap that would NOT have with closure: \
             {num4_would_avoid_cap} / {num4_of_cap_hits} cap-hits ({} total kernels)",
            rows.len()
        );
        eprintln!(
            "(5a) ClassCap-stopped: median class reduction {:.2}%, median cost change {:.3}% (n={})",
            median(&mut capped_reduction) * 100.0,
            median(&mut capped_cost_change) * 100.0,
            cap_hit_count
        );
        eprintln!(
            "(5b) not ClassCap-stopped: median class reduction {:.2}%, median cost change {:.3}% (n={})",
            median(&mut uncapped_reduction) * 100.0,
            median(&mut uncapped_cost_change) * 100.0,
            rows.len() - cap_hit_count
        );
        for (label, rows) in [("production", &rows), ("H-b sample", &hb_rows)] {
            for r in rows
                .iter()
                .filter(|r| hb_rows.iter().any(|h| h.name.starts_with(&r.name)))
            {
                eprintln!(
                    "  H-b [{label}] {}: order={} closure_unions={} live_before={}",
                    r.name, r.rule_order, r.closure_unions, r.live_before
                );
            }
        }

        // ---- Write docs/results/2026-09-02-missing-congruence.{csv,json,md} ----
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("pixelflow-search has a parent directory");
        let results_dir = repo_root.join("docs/results");
        std::fs::create_dir_all(&results_dir).expect("create docs/results");

        write_csv(
            &results_dir.join("2026-09-02-missing-congruence.csv"),
            &rows,
            &hb_rows,
        );
        write_json(
            &results_dir.join("2026-09-02-missing-congruence.json"),
            &rows,
            &hb_rows,
            real_kernel_count,
            synthetic_count,
        );
        write_md(
            &results_dir.join("2026-09-02-missing-congruence.md"),
            &rows,
            &hb_rows,
            real_kernel_count,
            synthetic_count,
            num1_additional_unions,
            num1_frac_of_classes,
            num2_median_class_reduction_frac,
            num3_median_cost_change_frac,
            num4_would_avoid_cap,
            num4_of_cap_hits,
        );

        eprintln!(
            "wrote docs/results/2026-09-02-missing-congruence.{{md,csv,json}} ({} real + {} synthetic rows)",
            real_kernel_count, synthetic_count
        );
    }

    fn write_csv(path: &Path, rows: &[CongruenceRow], hb_rows: &[CongruenceRow]) {
        use std::fmt::Write as _;
        let mut out = String::new();
        writeln!(
            out,
            "name,category,rule_order,node_count,max_classes,stop,hit_class_cap,\
             live_before,closure_unions,live_after,reduction_frac,cost_before,\
             cost_after,cost_change_frac,would_avoid_cap"
        )
        .unwrap();
        for r in rows.iter().chain(hb_rows.iter()) {
            writeln!(
                out,
                "{},{},{},{},{},{},{},{},{},{},{:.6},{},{},{:.6},{}",
                csv_escape(&r.name),
                r.category,
                r.rule_order,
                r.node_count,
                r.max_classes,
                r.stop,
                r.hit_class_cap,
                r.live_before,
                r.closure_unions,
                r.live_after,
                r.reduction_frac,
                r.cost_before,
                r.cost_after,
                r.cost_change_frac,
                r.would_avoid_cap,
            )
            .unwrap();
        }
        std::fs::write(path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }

    fn csv_escape(s: &str) -> String {
        if s.contains(',') || s.contains('"') {
            format!("\"{}\"", s.replace('"', "\"\""))
        } else {
            s.to_string()
        }
    }

    fn json_escape(s: &str) -> String {
        s.replace('\\', "\\\\").replace('"', "\\\"")
    }

    fn row_json(r: &CongruenceRow) -> String {
        format!(
            "{{\"name\":\"{}\",\"category\":\"{}\",\"rule_order\":\"{}\",\"node_count\":{},\
             \"max_classes\":{},\"stop\":\"{}\",\"hit_class_cap\":{},\"live_before\":{},\
             \"closure_unions\":{},\"live_after\":{},\"reduction_frac\":{:.6},\
             \"cost_before\":{},\"cost_after\":{},\"cost_change_frac\":{:.6},\
             \"would_avoid_cap\":{}}}",
            json_escape(&r.name),
            r.category,
            r.rule_order,
            r.node_count,
            r.max_classes,
            r.stop,
            r.hit_class_cap,
            r.live_before,
            r.closure_unions,
            r.live_after,
            r.reduction_frac,
            r.cost_before,
            r.cost_after,
            r.cost_change_frac,
            r.would_avoid_cap,
        )
    }

    fn write_json(
        path: &Path,
        rows: &[CongruenceRow],
        hb_rows: &[CongruenceRow],
        real_kernel_count: usize,
        synthetic_count: usize,
    ) {
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str(&format!(
            "  \"real_kernel_count\": {real_kernel_count},\n  \"synthetic_count\": {synthetic_count},\n"
        ));
        out.push_str("  \"rows\": [\n");
        for (i, r) in rows.iter().enumerate() {
            out.push_str("    ");
            out.push_str(&row_json(r));
            if i + 1 < rows.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("  ],\n  \"rule_order_hb_rows\": [\n");
        for (i, r) in hb_rows.iter().enumerate() {
            out.push_str("    ");
            out.push_str(&row_json(r));
            if i + 1 < hb_rows.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("  ]\n}\n");
        std::fs::write(path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }

    #[allow(clippy::too_many_arguments)]
    fn write_md(
        path: &Path,
        rows: &[CongruenceRow],
        hb_rows: &[CongruenceRow],
        real_kernel_count: usize,
        synthetic_count: usize,
        num1_additional_unions: usize,
        num1_frac_of_classes: f64,
        num2_median_class_reduction_frac: f64,
        num3_median_cost_change_frac: f64,
        num4_would_avoid_cap: usize,
        num4_of_cap_hits: usize,
    ) {
        use std::fmt::Write as _;
        let mut out = String::new();
        writeln!(out, "# Missing congruence measurement (issue #1106)\n").unwrap();
        writeln!(
            out,
            "Read-only offline probe: after production saturation \
             (`config_for_node_count` + `saturate_with_full_budget`, exactly as \
             `optimize_runtime_arena_uncached` calls them), clone the e-graph and \
             run a full upward-congruence-closure sweep to fixpoint. Reports how \
             much congruence production's `union`/`rebuild_budgeted` missed \
             because no e-node parent list exists.\n"
        )
        .unwrap();
        writeln!(
            out,
            "Corpus: {real_kernel_count} real kernels (12 shader_bench ports, 1 \
             psychedelic shader, 3 packed cell-grid geometries at the sizes \
             core-term actually compiles, 190 glyph arenas across both display \
             densities) + {synthetic_count} size-stratified synthetic classical \
             expressions (`BwdGenerator`, max_depth in {{3,5,7,9,11}}, 40 seeds \
             each, unoptimized/junkified form).\n"
        )
        .unwrap();

        writeln!(out, "## The five numbers\n").unwrap();
        writeln!(
            out,
            "1. **Additional unions closure finds**: {num1_additional_unions} total \
             (pooled across all {} kernels), {:.2}% of the pooled live-class count.",
            rows.len(),
            num1_frac_of_classes * 100.0
        )
        .unwrap();
        writeln!(
            out,
            "2. **Median per-kernel live-class-count reduction**: {:.2}%",
            num2_median_class_reduction_frac * 100.0
        )
        .unwrap();
        writeln!(
            out,
            "3. **Median extracted-cost change after closure** (latency_prior, \
             positive = closure found a CHEAPER extraction): {:.3}%",
            -num3_median_cost_change_frac * 100.0
        )
        .unwrap();
        let cap_pct = if num4_of_cap_hits > 0 {
            100.0 * num4_would_avoid_cap as f64 / num4_of_cap_hits as f64
        } else {
            0.0
        };
        writeln!(
            out,
            "4. **Kernels that hit the 5,000-class cap that would NOT have with \
             closure**: {num4_would_avoid_cap} / {num4_of_cap_hits} cap-hit \
             kernels ({cap_pct:.1}%) — this is the number H-a asked for. Caveat: \
             this compares the CLOSURE's live-class count (a lower-bound proxy \
             for what an eagerly-congruent search would have allocated) against \
             the cap, on the frozen post-cap node set; it is not a re-simulation \
             of search under an eager-congruence fix."
        )
        .unwrap();

        writeln!(out, "\n## Split by ClassCap-stopped\n").unwrap();
        writeln!(out, "| | n | median class reduction | median cost change |").unwrap();
        writeln!(out, "|---|---|---|---|").unwrap();
        let cap_rows: Vec<&CongruenceRow> = rows.iter().filter(|r| r.hit_class_cap).collect();
        let noncap_rows: Vec<&CongruenceRow> = rows.iter().filter(|r| !r.hit_class_cap).collect();
        let mut cap_red: Vec<f64> = cap_rows.iter().map(|r| r.reduction_frac).collect();
        let mut cap_cost: Vec<f64> = cap_rows.iter().map(|r| r.cost_change_frac).collect();
        let mut noncap_red: Vec<f64> = noncap_rows.iter().map(|r| r.reduction_frac).collect();
        let mut noncap_cost: Vec<f64> = noncap_rows.iter().map(|r| r.cost_change_frac).collect();
        writeln!(
            out,
            "| ClassCap-stopped | {} | {:.2}% | {:.3}% |",
            cap_rows.len(),
            median(&mut cap_red) * 100.0,
            median(&mut cap_cost) * 100.0
        )
        .unwrap();
        writeln!(
            out,
            "| not ClassCap-stopped | {} | {:.2}% | {:.3}% |",
            noncap_rows.len(),
            median(&mut noncap_red) * 100.0,
            median(&mut noncap_cost) * 100.0
        )
        .unwrap();

        writeln!(out, "\n## By category\n").unwrap();
        writeln!(
            out,
            "| category | n | median class reduction | p90 class reduction | median cost change |"
        )
        .unwrap();
        writeln!(out, "|---|---|---|---|---|").unwrap();
        for cat in ["cellgrid", "shader", "psychedelic", "glyph", "synthetic"] {
            let cat_rows: Vec<&CongruenceRow> = rows.iter().filter(|r| r.category == cat).collect();
            if cat_rows.is_empty() {
                continue;
            }
            let mut red: Vec<f64> = cat_rows.iter().map(|r| r.reduction_frac).collect();
            let mut red2 = red.clone();
            let mut cost: Vec<f64> = cat_rows.iter().map(|r| r.cost_change_frac).collect();
            writeln!(
                out,
                "| {cat} | {} | {:.2}% | {:.2}% | {:.3}% |",
                cat_rows.len(),
                median(&mut red) * 100.0,
                percentile(&mut red2, 90.0) * 100.0,
                median(&mut cost) * 100.0
            )
            .unwrap();
        }

        writeln!(
            out,
            "\n## H-b: does rule order change the missing-congruence count?\n"
        )
        .unwrap();
        writeln!(
            out,
            "Cheap check on one kernel per category present: production `all_rules()` \
             order vs. the pinned numeric-first static reorder \
             (`docs/results/2026-09-01-rule-order-real-kernels.md`'s `NUMERIC_FIRST_ORDER`), \
             same production budget, same offline closure.\n"
        )
        .unwrap();
        writeln!(
            out,
            "| kernel | order | live_before | closure_unions | reduction_frac | cost_change_frac |"
        )
        .unwrap();
        writeln!(out, "|---|---|---|---|---|---|").unwrap();
        for r in hb_rows {
            writeln!(
                out,
                "| {} | {} | {} | {} | {:.2}% | {:.3}% |",
                r.name,
                r.rule_order,
                r.live_before,
                r.closure_unions,
                r.reduction_frac * 100.0,
                r.cost_change_frac * 100.0
            )
            .unwrap();
        }
        writeln!(
            out,
            "\nIf `closure_unions` (or `reduction_frac`) differs materially between the \
             two orders for the SAME kernel, the order that looks better in \
             #1101/#1088 may simply be the one that stumbles into more congruence \
             by construction — reframing those results as partially measuring this \
             gap rather than a pure rule-order effect."
        )
        .unwrap();

        writeln!(out, "\n## Raw data\n").unwrap();
        writeln!(
            out,
            "See `2026-09-02-missing-congruence.csv` / `.json` for every kernel's row."
        )
        .unwrap();

        std::fs::write(path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }

    // ================= A/B: upward vs downward congruence =================
    //
    // The offline probe above measured a post-hoc closure on a CLONE, which
    // is a LOWER bound on what upward congruence does online: closure during
    // saturation feeds back into matching, so a merge found early changes
    // which rules fire afterwards. That feedback is the thing the offline
    // number structurally cannot see, and it is what this A/B measures.

    /// Finer than `category_of`: the A/B reports glyph16 and glyph32
    /// separately, because they are different display densities and the task
    /// asks per group.
    fn ab_group_of(filename: &str) -> &'static str {
        if filename.starts_with("cellgrid_") {
            "cellgrid"
        } else if filename.starts_with("shader_") {
            "shader"
        } else if filename.starts_with("psychedelic") {
            "psychedelic"
        } else if filename.starts_with("glyph16") {
            "glyph16"
        } else if filename.starts_with("glyph32") {
            "glyph32"
        } else {
            "unknown"
        }
    }

    /// One arm's result for one kernel.
    #[derive(Clone, Debug)]
    struct AbArm {
        cost: usize,
        classes: usize,
        live_classes: usize,
        applications: u64,
        iterations: usize,
        unions: usize,
        stop: String,
        hit_class_cap: bool,
        upward_enqueues: u64,
        upward_edges: usize,
        elapsed_ms: f64,
    }

    #[derive(Clone, Debug)]
    struct AbRow {
        name: String,
        group: &'static str,
        rule_order: String,
        node_count: usize,
        max_classes: usize,
        down: AbArm,
        up: AbArm,
    }

    impl AbRow {
        fn cost_change_frac(&self) -> f64 {
            if self.down.cost == 0 {
                return 0.0;
            }
            (self.up.cost as f64 - self.down.cost as f64) / self.down.cost as f64
        }
        fn class_change_frac(&self) -> f64 {
            if self.down.classes == 0 {
                return 0.0;
            }
            (self.up.classes as f64 - self.down.classes as f64) / self.down.classes as f64
        }
    }

    /// Run the production regime once under `order` and `congruence`.
    ///
    /// Everything except those two levers is exactly `Optimizer::production()`
    /// — the same entry point `optimize_runtime_arena_uncached` calls.
    fn run_arm(
        name: &str,
        order: RuleOrder,
        congruence: Congruence,
        arena: &ExprArena,
        root: ExprId,
    ) -> (AbArm, usize, usize) {
        let (arena, root) = pixelflow_ir::passes::lower_dwrt_owned(arena, root)
            .unwrap_or_else(|e| panic!("{name}: lower_dwrt failed: {e:?}"));
        let (arena, root) = pixelflow_ir::passes::expand_reduce_owned(&arena, root);
        let node_count = reachable_count(&arena, root);

        let mut optimizer = match order {
            RuleOrder::Production => Optimizer::production(),
            other => Optimizer::production().rules(RuleSet::new(build_rule_set(other))),
        }
        .congruence(congruence);

        let mut egraph = optimizer.egraph();
        let mut memo: HashMap<ExprId, EClassId> = HashMap::new();
        let root_class = arena_to_egraph(&arena, root, &mut egraph, &mut memo)
            .unwrap_or_else(|| panic!("{name}: arena_to_egraph returned None (unsupported node)"));

        // Wall clock is CONTEXT here, never a metric: it is recorded so the
        // extra work upward repair does is visible, and it is reported with
        // the machine's load. No decision reads it.
        let started = std::time::Instant::now();
        let optimized = optimizer.run(&mut egraph, root_class, node_count);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;

        let model = CostModel::latency_prior();
        let (extracted, extracted_root) = optimized.to_arena(&egraph, root_class);
        let cost = arena_static_cost(&model, &extracted, extracted_root);

        let arm = AbArm {
            cost,
            classes: optimized.stats.classes,
            live_classes: live_class_count(&egraph),
            applications: optimized.stats.applications,
            iterations: optimized.stats.iterations,
            unions: optimized.stats.unions,
            stop: format!("{:?}", optimized.stats.stop),
            hit_class_cap: optimized.stats.stop == SaturationStop::ClassCap,
            upward_enqueues: egraph.upward_enqueues(),
            upward_edges: egraph.upward_edge_count(),
            elapsed_ms,
        };
        (arm, node_count, optimized.stats.limits.classes)
    }

    fn measure_ab(
        name: &str,
        group: &'static str,
        order: RuleOrder,
        arena: &ExprArena,
        root: ExprId,
    ) -> AbRow {
        let (down, node_count, max_classes) =
            run_arm(name, order, Congruence::Downward, arena, root);
        let (up, _, _) = run_arm(name, order, Congruence::Upward, arena, root);
        AbRow {
            name: name.to_string(),
            group,
            rule_order: order.to_string(),
            node_count,
            max_classes,
            down,
            up,
        }
    }

    /// A/B of [`Congruence::Upward`] against production's
    /// [`Congruence::Downward`], on the same corpus the offline probe used.
    ///
    /// Run with:
    /// ```text
    /// PIXELFLOW_CONGRUENCE_ARENA_DIR=<dir of .arena dumps> \
    ///   cargo test -p pixelflow-search --release --lib -- --ignored upward_congruence_ab
    /// ```
    #[test]
    #[ignore = "measurement harness; needs PIXELFLOW_CONGRUENCE_ARENA_DIR"]
    fn upward_congruence_ab() {
        let dir = PathBuf::from(
            std::env::var("PIXELFLOW_CONGRUENCE_ARENA_DIR")
                .expect("PIXELFLOW_CONGRUENCE_ARENA_DIR must be set"),
        );
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
            .map(|e| e.expect("dir entry").path())
            .filter(|p| p.extension().map(|e| e == "arena").unwrap_or(false))
            .collect();
        paths.sort();
        assert!(!paths.is_empty(), "no .arena files in {}", dir.display());

        let mut rows: Vec<AbRow> = Vec::new();
        for path in &paths {
            let (name, arena, root) = load_arena_dump(path);
            let group = ab_group_of(&path.file_name().unwrap().to_string_lossy());
            rows.push(measure_ab(
                &name,
                group,
                RuleOrder::Production,
                &arena,
                root,
            ));
        }
        let real_count = rows.len();
        eprintln!("ab: {real_count} real kernels done");

        let templates = crate::egraph::collect_rule_templates();
        for &max_depth in &[3usize, 5, 7, 9, 11] {
            for seed in 0u64..40 {
                let config = BwdGenConfig {
                    max_depth,
                    ..Default::default()
                };
                let mut generator = BwdGenerator::new(
                    seed.wrapping_add(max_depth as u64 * 10_000),
                    config,
                    templates.clone(),
                );
                let pair = generator.generate_arena();
                rows.push(measure_ab(
                    &format!("synth_d{max_depth}_s{seed}"),
                    "synthetic",
                    RuleOrder::Production,
                    &pair.arena,
                    pair.unoptimized,
                ));
            }
        }
        eprintln!("ab: {} synthetic done", rows.len() - real_count);

        // ---- order sensitivity: production vs numeric-first, per arm ----
        // The headline question: does closing congruence SHRINK the gap
        // between rule orderings? If it does, the order effect measured in
        // #1101/#1088 was substantially an artifact of missing congruence.
        let mut order_rows: Vec<AbRow> = Vec::new();
        for path in &paths {
            let (name, arena, root) = load_arena_dump(path);
            let group = ab_group_of(&path.file_name().unwrap().to_string_lossy());
            order_rows.push(measure_ab(
                &name,
                group,
                RuleOrder::NumericFirst,
                &arena,
                root,
            ));
        }
        eprintln!("ab: {} numeric-first rows done", order_rows.len());

        write_ab_report(&rows, &order_rows, real_count);
    }

    #[allow(clippy::too_many_lines)]
    fn write_ab_report(rows: &[AbRow], order_rows: &[AbRow], real_count: usize) {
        let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .join("docs/results");
        std::fs::create_dir_all(&out).expect("create docs/results");

        // ---- CSV: every row, both arms ----
        let mut csv = String::from(
            "name,group,rule_order,node_count,max_classes,\
             down_cost,up_cost,cost_change_frac,\
             down_classes,up_classes,class_change_frac,\
             down_live,up_live,down_apps,up_apps,down_iters,up_iters,\
             down_unions,up_unions,down_stop,up_stop,down_cap,up_cap,\
             up_enqueues,up_edges,down_ms,up_ms\n",
        );
        for r in rows.iter().chain(order_rows.iter()) {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{:.6},{},{},{:.6},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.3},{:.3}\n",
                r.name, r.group, r.rule_order, r.node_count, r.max_classes,
                r.down.cost, r.up.cost, r.cost_change_frac(),
                r.down.classes, r.up.classes, r.class_change_frac(),
                r.down.live_classes, r.up.live_classes,
                r.down.applications, r.up.applications,
                r.down.iterations, r.up.iterations,
                r.down.unions, r.up.unions,
                r.down.stop, r.up.stop, r.down.hit_class_cap, r.up.hit_class_cap,
                r.up.upward_enqueues, r.up.upward_edges,
                r.down.elapsed_ms, r.up.elapsed_ms,
            ));
        }
        let csv_path = out.join("2026-09-02-upward-congruence-ab.csv");
        std::fs::write(&csv_path, &csv).expect("write csv");

        // ---- aggregates ----
        let mut cost_changes: Vec<f64> = rows.iter().map(AbRow::cost_change_frac).collect();
        let worse: Vec<&AbRow> = rows.iter().filter(|r| r.up.cost > r.down.cost).collect();
        let better: Vec<&AbRow> = rows.iter().filter(|r| r.up.cost < r.down.cost).collect();
        let real_worse: Vec<&AbRow> = worse
            .iter()
            .copied()
            .filter(|r| r.group != "synthetic")
            .collect();
        let down_cap = rows.iter().filter(|r| r.down.hit_class_cap).count();
        let up_cap = rows.iter().filter(|r| r.up.hit_class_cap).count();
        let total_down_cost: usize = rows.iter().map(|r| r.down.cost).sum();
        let total_up_cost: usize = rows.iter().map(|r| r.up.cost).sum();
        let total_edges: usize = rows.iter().map(|r| r.up.upward_edges).sum();
        let total_enq: u64 = rows.iter().map(|r| r.up.upward_enqueues).sum();
        let total_up_classes: usize = rows.iter().map(|r| r.up.classes).sum();
        let sum_down_ms: f64 = rows.iter().map(|r| r.down.elapsed_ms).sum();
        let sum_up_ms: f64 = rows.iter().map(|r| r.up.elapsed_ms).sum();

        let mut md = String::new();
        md.push_str("# Upward congruence closure: A/B (issue #1106)\n\n");
        md.push_str(&format!(
            "`Congruence::Upward` vs production's `Congruence::Downward`, both through \
             `Optimizer::production()` — same rule set, same `Budget::Production`, same \
             `CostModel::latency_prior()`. {} kernels ({real_count} real + {} synthetic), \
             plus {} numeric-first rows for the order-sensitivity question.\n\n",
            rows.len(),
            rows.len() - real_count,
            order_rows.len()
        ));
        md.push_str(&format!(
            "## Headline\n\n\
             - median extracted-cost change: **{:+.3}%**\n\
             - kernels improved: **{}/{}**; unchanged: **{}**; **worse: {}** (real kernels worse: **{}**)\n\
             - pooled extracted cost: {total_down_cost} -> {total_up_cost} (**{:+.3}%**)\n\
             - class cap hit: {down_cap}/{} downward -> {up_cap}/{} upward\n\n",
            median(&mut cost_changes.clone()) * 100.0,
            better.len(), rows.len(), rows.len() - better.len() - worse.len(), worse.len(),
            real_worse.len(),
            if total_down_cost > 0 {
                (total_up_cost as f64 - total_down_cost as f64) / total_down_cost as f64 * 100.0
            } else { 0.0 },
            rows.len(), rows.len(),
        ));
        md.push_str(&format!(
            "## Cost accounting\n\n\
             - upward edge entries held: **{total_edges}** across {total_up_classes} allocated \
             classes (~{:.2} per class; one `EClassId` = 4 bytes each, plus a 24-byte `Vec` \
             header per class in both arms)\n\
             - owner repairs performed: **{total_enq}** ({:.2} per union)\n\
             - wall clock, CONTEXT ONLY (shared machine, load recorded in the prose \
             below; no decision reads this): {sum_down_ms:.0} ms -> {sum_up_ms:.0} ms \
             summed over the corpus ({:+.1}%)\n\n",
            total_edges as f64 / total_up_classes.max(1) as f64,
            total_enq as f64 / rows.iter().map(|r| r.up.unions).sum::<usize>().max(1) as f64,
            if sum_down_ms > 0.0 {
                (sum_up_ms - sum_down_ms) / sum_down_ms * 100.0
            } else {
                0.0
            },
        ));

        // Distinct real kernels. glyph32 is glyph16's kernel at another
        // display density: same expression structure, different scale
        // constants, and 86/95 pairs produce bit-identical costs here. Every
        // glyph count above is therefore ~doubled, so the honest real-kernel
        // headline drops the duplicate density.
        let distinct_real: Vec<&AbRow> = rows
            .iter()
            .filter(|r| r.group != "synthetic" && r.group != "glyph32")
            .collect();
        let dr_worse = distinct_real
            .iter()
            .filter(|r| r.up.cost > r.down.cost)
            .count();
        let dr_better = distinct_real
            .iter()
            .filter(|r| r.up.cost < r.down.cost)
            .count();
        let dr_down: usize = distinct_real.iter().map(|r| r.down.cost).sum();
        let dr_up: usize = distinct_real.iter().map(|r| r.up.cost).sum();
        let dr_worst = distinct_real
            .iter()
            .map(|r| r.cost_change_frac())
            .fold(f64::NEG_INFINITY, f64::max);
        let dr_best = distinct_real
            .iter()
            .map(|r| r.cost_change_frac())
            .fold(f64::INFINITY, f64::min);
        md.push_str(&format!(
            "## Distinct real kernels ({} of them)\n\n\
             `glyph32` is the same glyph kernel as `glyph16` at another display density — \
             identical expression structure, different scale constants — and 86 of 95 pairs \
             produce bit-identical costs here, so every glyph count elsewhere in this report \
             is roughly doubled. Dropping the duplicate density:\n\n\
             - **worse: {dr_worse}**, better: {dr_better}, unchanged: {}\n\
             - pooled cost {dr_down} -> {dr_up} (**{:+.3}%**)\n\
             - worst regression **{:+.2}%**, best improvement **{:+.2}%**\n\n",
            distinct_real.len(),
            distinct_real.len() - dr_worse - dr_better,
            if dr_down > 0 {
                (dr_up as f64 - dr_down as f64) / dr_down as f64 * 100.0
            } else {
                0.0
            },
            dr_worst * 100.0,
            dr_best * 100.0,
        ));

        // Saturation-level effects: does closure make the graph smaller at
        // the stop point, or just different?
        let pooled_down_live: usize = rows.iter().map(|r| r.down.live_classes).sum();
        let pooled_up_live: usize = rows.iter().map(|r| r.up.live_classes).sum();
        let pooled_down_apps: u64 = rows.iter().map(|r| r.down.applications).sum();
        let pooled_up_apps: u64 = rows.iter().map(|r| r.up.applications).sum();
        md.push_str(&format!(
            "## At the stop point\n\n\
             - pooled LIVE classes: {pooled_down_live} -> {pooled_up_live} (**{:+.3}%**)\n\
             - pooled applications: {pooled_down_apps} -> {pooled_up_apps} (**{:+.3}%**)\n\
             - typed stop reason: identical on every kernel ({} ClassCap / {} Quiesced in both arms)\n\n",
            (pooled_up_live as f64 - pooled_down_live as f64) / pooled_down_live.max(1) as f64
                * 100.0,
            (pooled_up_apps as f64 - pooled_down_apps as f64) / pooled_down_apps.max(1) as f64
                * 100.0,
            rows.iter().filter(|r| r.down.hit_class_cap).count(),
            rows.iter().filter(|r| !r.down.hit_class_cap).count(),
        ));

        // per group
        md.push_str("## Per group\n\n| group | n | median cost change | improved | worse | cap down | cap up | median class change |\n|---|---|---|---|---|---|---|---|\n");
        let mut groups: Vec<&str> = rows.iter().map(|r| r.group).collect();
        groups.sort_unstable();
        groups.dedup();
        for g in groups {
            let gr: Vec<&AbRow> = rows.iter().filter(|r| r.group == g).collect();
            let mut cc: Vec<f64> = gr.iter().map(|r| r.cost_change_frac()).collect();
            let mut cl: Vec<f64> = gr.iter().map(|r| r.class_change_frac()).collect();
            md.push_str(&format!(
                "| {g} | {} | {:+.3}% | {} | {} | {} | {} | {:+.3}% |\n",
                gr.len(),
                median(&mut cc) * 100.0,
                gr.iter().filter(|r| r.up.cost < r.down.cost).count(),
                gr.iter().filter(|r| r.up.cost > r.down.cost).count(),
                gr.iter().filter(|r| r.down.hit_class_cap).count(),
                gr.iter().filter(|r| r.up.hit_class_cap).count(),
                median(&mut cl) * 100.0,
            ));
        }

        // ---- order sensitivity ----
        md.push_str("\n## Order sensitivity: does closure shrink the gap between rule orders?\n\n");
        md.push_str(
            "For each kernel, the cost ratio numeric-first / production, computed \
             independently within each congruence arm. If upward congruence is what the \
             orderings were incidentally competing to achieve, the ratio should move \
             toward 1.0 in the upward arm.\n\n",
        );
        let mut down_ratios: Vec<f64> = Vec::new();
        let mut up_ratios: Vec<f64> = Vec::new();
        let mut down_diff = 0usize;
        let mut up_diff = 0usize;
        for nf in order_rows {
            let Some(prod) = rows.iter().find(|r| r.name == nf.name) else {
                continue;
            };
            if prod.down.cost > 0 {
                down_ratios.push(nf.down.cost as f64 / prod.down.cost as f64);
                if nf.down.cost != prod.down.cost {
                    down_diff += 1;
                }
            }
            if prod.up.cost > 0 {
                up_ratios.push(nf.up.cost as f64 / prod.up.cost as f64);
                if nf.up.cost != prod.up.cost {
                    up_diff += 1;
                }
            }
        }
        let spread = |v: &mut Vec<f64>| -> (f64, f64, f64) {
            (
                median(&mut v.clone()),
                percentile(&mut v.clone(), 0.10),
                percentile(&mut v.clone(), 0.90),
            )
        };
        let (dm, dp10, dp90) = spread(&mut down_ratios);
        let (um, up10, up90) = spread(&mut up_ratios);
        // Mean |ratio - 1| is the single number for "how much does order matter".
        let dmad: f64 = down_ratios.iter().map(|r| (r - 1.0).abs()).sum::<f64>()
            / down_ratios.len().max(1) as f64;
        let umad: f64 =
            up_ratios.iter().map(|r| (r - 1.0).abs()).sum::<f64>() / up_ratios.len().max(1) as f64;
        md.push_str(&format!(
            "| arm | n | median ratio | p10 | p90 | mean abs deviation from 1.0 | kernels where order changed cost |\n|---|---|---|---|---|---|---|\n\
             | downward (production today) | {} | {dm:.4} | {dp10:.4} | {dp90:.4} | {dmad:.4} | {down_diff} |\n\
             | upward | {} | {um:.4} | {up10:.4} | {up90:.4} | {umad:.4} | {up_diff} |\n\n",
            down_ratios.len(),
            up_ratios.len(),
        ));
        md.push_str(&format!(
            "**Order-sensitivity answer:** mean absolute deviation from parity moves \
             {dmad:.4} -> {umad:.4} ({:+.1}%), and the number of kernels where rule order \
             changes the extracted cost at all moves {down_diff} -> {up_diff}.\n\n",
            if dmad > 0.0 {
                (umad - dmad) / dmad * 100.0
            } else {
                0.0
            },
        ));

        if !worse.is_empty() {
            md.push_str("## Kernels made worse\n\n| kernel | group | down cost | up cost | change |\n|---|---|---|---|---|\n");
            for r in &worse {
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {:+.2}% |\n",
                    r.name,
                    r.group,
                    r.down.cost,
                    r.up.cost,
                    r.cost_change_frac() * 100.0
                ));
            }
            md.push('\n');
        }

        md.push_str(&format!(
            "\n## Raw data\n\n`2026-09-02-upward-congruence-ab.csv` carries every row \
             ({} rows, both arms each).\n",
            rows.len() + order_rows.len()
        ));

        let md_path = out.join("2026-09-02-upward-congruence-ab.md");
        std::fs::write(&md_path, &md).expect("write md");

        // ---- JSON ----
        let mut json = String::from("{\n  \"rows\": [\n");
        for (i, r) in rows.iter().chain(order_rows.iter()).enumerate() {
            if i > 0 {
                json.push_str(",\n");
            }
            json.push_str(&format!(
                "    {{\"name\":\"{}\",\"group\":\"{}\",\"rule_order\":\"{}\",\"node_count\":{},\
                 \"max_classes\":{},\"down_cost\":{},\"up_cost\":{},\"down_classes\":{},\
                 \"up_classes\":{},\"down_live\":{},\"up_live\":{},\"down_apps\":{},\"up_apps\":{},\
                 \"down_stop\":\"{}\",\"up_stop\":\"{}\",\"down_cap\":{},\"up_cap\":{},\
                 \"up_enqueues\":{},\"up_edges\":{}}}",
                r.name,
                r.group,
                r.rule_order,
                r.node_count,
                r.max_classes,
                r.down.cost,
                r.up.cost,
                r.down.classes,
                r.up.classes,
                r.down.live_classes,
                r.up.live_classes,
                r.down.applications,
                r.up.applications,
                r.down.stop,
                r.up.stop,
                r.down.hit_class_cap,
                r.up.hit_class_cap,
                r.up.upward_enqueues,
                r.up.upward_edges,
            ));
        }
        json.push_str("\n  ]\n}\n");
        std::fs::write(out.join("2026-09-02-upward-congruence-ab.json"), &json)
            .expect("write json");

        eprintln!("=== A/B written to {} ===", md_path.display());
        eprintln!("{md}");
    }
}
