//! E-graph optimization for runtime-built kernels.
//!
//! `kernel!` already runs full saturation at macro-expansion
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

use crate::egraph::{EClassId, EGraph, ENode, Optimizer};
use crate::saturate_pass::Saturate;
use pixelflow_ir::LatticeShape;
use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{BufferDecl, ExprArena, ExprId, ExprNode};
use pixelflow_ir::optimize::{Identity, Optimize};
use pixelflow_ir::passes::{ExpandReduce, ExpandRefs, LowerDwrt};
use pixelflow_ir::pipeline;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Optimize a runtime-built arena via bounded e-graph saturation, through
/// the same [`Optimizer`] entry point — rule set, budget, cost model,
/// extractor — as the `kernel!` macro.
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
/// `Uniform` leaves are representable like `Buffer`: opaque to every rule,
/// hash-consed by identity, redeclared by extraction. Nothing folds one.
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
/// back `Arc<CompiledKernel>` rather than owned code: a hit must be an atomic
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
    //
    // Uniform-bearing arenas bypass it for the same reason and one more: the
    // optimized arena carries its uniforms' identities, and the link step
    // downstream maps *those* to block offsets. A hit keyed on structure
    // alone would hand a second composition an arena naming the first one's
    // instances. The JIT cache in front of this one is keyed on structure
    // (dense offsets, not identities), so the saturation is still paid once
    // per shape.
    if !arena.buffers().is_empty() || !arena.uniforms().is_empty() {
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
    key.push(saturation_switch() as u8);
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
    // The tier's pipeline, as a composition rather than three hand-sequenced
    // calls. The order is load-bearing and is now the expression itself:
    //
    // `ExpandRefs` first, because every step after it reads structure and a
    // reference has none to read — its body is in the `KernelStore`, not in
    // this arena. It is an identity when nothing composed by reference, which
    // is every kernel production builds today.
    //
    // `Saturate` next, on the arena as *written* — folds still folded,
    // derivatives still `Dwrt`. Both are things the graph knows: the chain
    // rule and a fold's decompositions are rule sets (`egraph::derivative`,
    // `egraph::fold_rules`), so the graph resolves what it judges worth
    // resolving and keeps the rest folded.
    //
    // **`LowerDwrt` and `ExpandReduce` come last, and that is the whole
    // point.** Legalization is the *fallback*: it takes whatever illegal
    // shape survived saturation — a `Dwrt` the chain rule did not reach, a
    // `Reduce` the graph declined to peel — and makes it emittable. It owns
    // nothing the graph does not also know, so running it first only takes
    // choices away.
    //
    // Size is not symmetric across that boundary, which is why the order is
    // not arbitrary. Unrolling a 34-piece fold *before* saturation hands the
    // e-graph 141,530 nodes for one glyph, and an e-graph is quadratic-ish in
    // what it is fed: that is where the budget goes. Unrolling it *after*
    // hands the same expansion to the assembler, which is linear and does not
    // care — a million-node IR is a routine afternoon for a register
    // allocator and a catastrophe for saturation. So the emitted node count
    // rises when the legalizer moves back, and that is the trade being made,
    // not a regression: the e-graph gets to see the small, high-level program
    // it can actually reason about.
    //
    // A declining step short-circuits the rest and yields `None` here, which
    // means exactly what it always meant: the caller compiles its own arena
    // unchanged, unoptimized but correct — and legalizes it itself, since
    // `Manifold::compile` runs `passes::legalize` regardless of whether this
    // function returned anything.
    match saturation_switch() {
        SaturationSwitch::On => pipeline![
            ExpandRefs,
            Saturate::runtime(shape),
            LowerDwrt,
            ExpandReduce
        ]
        .optimize(arena, root)
        .into_changed(),
        // The `Identity` path: the same legalizing tail, no saturation.
        // What `Lattice::bake` would emit if the e-graph did not exist —
        // the "F" column of docs/plans/2026-09-06-egraph-at-production-scale.md
        // §7, measured by docs/results/2026-09-07-egraph-off-vs-on-real-shaders.md.
        SaturationSwitch::Off => pipeline![ExpandRefs, Identity, LowerDwrt, ExpandReduce]
            .optimize(arena, root)
            .into_changed(),
    }
}

/// Whether the runtime tier saturates at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum SaturationSwitch {
    Off = 0,
    On = 1,
}

/// The one place `PIXELFLOW_SATURATION` is read.
///
/// `off` selects the `Identity` path above; `on` or unset selects
/// saturation; any other value is a hard error. The variable is honoured
/// only under the `saturation-switch` cargo feature (a measurement build:
/// `pixelflow-pipeline`'s `egraph_off_on` harness). A build without the
/// feature panics if the variable is set at all, so an `export
/// PIXELFLOW_SATURATION=off` left behind in a shell can never quietly ship
/// unoptimized kernels — the switch is not leavable-on by accident.
fn saturation_switch() -> SaturationSwitch {
    static SWITCH: OnceLock<SaturationSwitch> = OnceLock::new();
    *SWITCH.get_or_init(|| {
        let var = std::env::var("PIXELFLOW_SATURATION");
        #[cfg(not(feature = "saturation-switch"))]
        {
            assert!(
                matches!(var, Err(std::env::VarError::NotPresent)),
                "PIXELFLOW_SATURATION is set ({var:?}) but this build has no \
                 `saturation-switch` feature (pixelflow-search); the variable is a \
                 measurement switch and a production build refuses to guess what \
                 it means. Unset it."
            );
            SaturationSwitch::On
        }
        #[cfg(feature = "saturation-switch")]
        match var.as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("on") => SaturationSwitch::On,
            Ok("off") => SaturationSwitch::Off,
            other => panic!("PIXELFLOW_SATURATION must be `on` or `off` (or unset), got {other:?}"),
        }
    })
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
    let len = arena.len();
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
            &ExprNode::Uniform(u) => {
                // Keyed by identity for the reason `Buffer` is; unreachable
                // in practice, since uniform-bearing arenas bypass the cache.
                key.push(8);
                let decl = *arena.uniform_decl(u);
                key.extend_from_slice(alloc::format!("{:?}", decl.id).as_bytes());
                key.extend_from_slice(&decl.default.to_bits().to_le_bytes());
            }
            // A reference's key IS its referent's content, digested, so
            // encoding the key is encoding the kernel it names.
            &ExprNode::Ref(k) => {
                key.push(9);
                key.extend_from_slice(&k.bits().to_le_bytes());
            }
            // The fold is metadata, so it goes in the key's tag bytes rather
            // than as three child nodes. Two folds over one body under
            // different algebras, binders or ranges must not share a cache
            // entry, and this is where that is said.
            &ExprNode::Reduce { fold, body } => {
                key.push(10);
                key.extend_from_slice(&fold.to_bits().to_le_bytes());
                push_id(&mut key, &dense, body);
            }
            // Same shape as `pixelflow_ir::key`'s canonical form: the mask
            // densifies as a child, `on`/`off` are content-addressed and go
            // in directly.
            &ExprNode::Guard { mask, on, off } => {
                key.push(11);
                push_id(&mut key, &dense, mask);
                key.extend_from_slice(&on.bits().to_le_bytes());
                key.extend_from_slice(&off.bits().to_le_bytes());
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
            &ExprNode::Nary(op, _start, n) => {
                key.push(7);
                key.extend_from_slice(&op.marshal().to_bytes());
                key.extend_from_slice(&n.to_le_bytes());
                for child in arena.children(id) {
                    push_id(&mut key, &dense, child);
                }
            }
        }
        dense[idx] = next;
        next += 1;
    }

    key
}

/// Whether the runtime tier can represent `kind` in its e-graph — i.e.,
/// whether an arena containing it still optimizes rather than bailing.
/// Test hook for the representability guards; the semantics live in
/// [`Vocabulary::Runtime`](crate::egraph::Vocabulary).
#[must_use]
pub fn is_egraph_representable(kind: OpKind) -> bool {
    crate::egraph::Vocabulary::Runtime.resolve(kind).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::OpKind;
    use pixelflow_ir::arena::BufferDecl;
    use pixelflow_ir::binding::BindingTable;
    use pixelflow_ir::fold::{Binder, Fold, Monoid};

    #[test]
    fn saturation_switch_follows_the_variable() {
        use super::SaturationSwitch;
        // Without the feature a set variable is a panic (loud, in the call
        // below); with it, the mapping is the contract.
        #[cfg(not(feature = "saturation-switch"))]
        let expected = SaturationSwitch::On;
        #[cfg(feature = "saturation-switch")]
        let expected = match std::env::var("PIXELFLOW_SATURATION").as_deref() {
            Err(_) | Ok("on") => SaturationSwitch::On,
            Ok("off") => SaturationSwitch::Off,
            Ok(other) => panic!("unexpected PIXELFLOW_SATURATION={other:?} in a test process"),
        };
        assert_eq!(super::saturation_switch(), expected);
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

    /// Whether any `Nary` (the `Reduce` binder) is reachable from `root`.
    /// Whether a binder survives to the optimized arena. Named for what it
    /// asks rather than for the node shape it used to look for: a fold is
    /// `ExprNode::Reduce` now, not an `Nary`.
    fn reaches_a_binder(arena: &ExprArena, root: ExprId) -> bool {
        let mut seen = vec![false; arena.nodes_raw().len()];
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            if matches!(arena.node(id), ExprNode::Nary(..) | ExprNode::Reduce { .. }) {
                return true;
            }
            stack.extend(arena.children(id));
        }
        false
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
    use crate::arena_corpus::{category_of, load_arena_dump, median, percentile};
    use crate::egraph::rule_order::{RuleOrder, build_rule_set};
    use crate::egraph::{CostModel, RuleSet, SaturationStop, choices_to_arena};
    use crate::nnue::{BwdGenConfig, BwdGenerator};
    use std::path::{Path, PathBuf};

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
    /// `ExtractedDAG::total_cost`, which is a TREE cost and prices a shared
    /// subterm once per use. Since #1111 `ExtractedDAG::dag_cost` is the same
    /// quantity this computes; the arena walk is kept as the independent
    /// check that they agree.
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

    /// The `.arena` dumps of the real-kernel corpus, sorted, from
    /// `PIXELFLOW_CONGRUENCE_ARENA_DIR`. Every probe in this module reads
    /// the same corpus the same way; three transcriptions of this would be
    /// three chances for two probes to silently measure different kernels.
    fn arena_corpus_paths() -> Vec<PathBuf> {
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
        paths
    }

    /// One kernel's post-saturation e-graph, plus the numbers every probe
    /// over this corpus needs from it. Owning the graph (rather than the
    /// `SaturationResult`) is what lets a caller clone it and experiment.
    struct ProductionRun {
        egraph: EGraph,
        root_class: EClassId,
        node_count: usize,
        max_classes: usize,
        stop: SaturationStop,
        /// `arena_static_cost` of the extraction production would actually
        /// emit for this kernel.
        cost: usize,
        /// Nodes reachable from the extracted arena's root — the emitted
        /// kernel's size, as a second observable alongside its price. Two
        /// runs agreeing on `cost` but not on this would mean the cost
        /// model happened to tie, not that the same kernel came out.
        extracted_nodes: usize,
    }

    /// Run the production regime — `config_for_node_count` +
    /// `saturate_with_full_budget`, exactly as
    /// `optimize_runtime_arena_uncached` calls them — under rule order
    /// `order`, and extract. Every probe in this module goes through here:
    /// a second transcription of the production call sequence is a future
    /// divergence, and the whole point of these measurements is that they
    /// measure production.
    fn run_production(
        name: &str,
        order: RuleOrder,
        arena: &ExprArena,
        root: ExprId,
    ) -> ProductionRun {
        // What `optimize_runtime_arena_uncached` hands the e-graph:
        // `ExpandRefs` and nothing else. `LowerDwrt`/`ExpandReduce` run after
        // saturation — a `Dwrt` and a `Reduce` are both things the rule set
        // knows, and legalization is the fallback for what it declined — so
        // lowering here would measure a pipeline that no longer exists.
        let (arena, root) = pixelflow_ir::passes::expand_refs_owned(arena, root);
        let node_count = crate::egraph::reachable_count(&arena, root);

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
        let root_class = crate::egraph::insert(
            &arena,
            root,
            &mut egraph,
            crate::egraph::Vocabulary::Runtime,
        )
        .unwrap_or_else(|e| panic!("{name}: insert declined ({e:?})"));

        let optimized = optimizer.run(&mut egraph, root_class, node_count);
        let max_classes = optimized.stats.limits.classes;

        let model = CostModel::latency_prior();
        let (extracted, extracted_root) = optimized.to_arena(&egraph, root_class);
        let cost = arena_static_cost(&model, &extracted, extracted_root);
        let extracted_nodes = crate::egraph::reachable_count(&extracted, extracted_root);

        ProductionRun {
            egraph,
            root_class,
            node_count,
            max_classes,
            stop: optimized.stats.stop,
            cost,
            extracted_nodes,
        }
    }

    /// Run the production regime under rule order `order`, then measure the
    /// offline upward-closure gap on a clone.
    fn measure_one(
        name: &str,
        category: &'static str,
        order: RuleOrder,
        arena: &ExprArena,
        root: ExprId,
    ) -> CongruenceRow {
        let ProductionRun {
            egraph,
            root_class,
            node_count,
            max_classes,
            stop,
            cost: cost_before,
            ..
        } = run_production(name, order, arena, root);
        let hit_class_cap = stop == SaturationStop::ClassCap;
        let live_before = live_class_count(&egraph);
        let model = CostModel::latency_prior();

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
            stop: format!("{stop:?}"),
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
        let paths = arena_corpus_paths();
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
        let all_reduction_frac: Vec<f64> = rows.iter().map(|r| r.reduction_frac).collect();
        let all_cost_change_frac: Vec<f64> = rows.iter().map(|r| r.cost_change_frac).collect();
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

    /// One kernel's production-regime numbers under one build of
    /// `rebuild_budgeted` — the unit the write-back A/B diffs.
    #[derive(Clone, Debug)]
    struct RepairRow {
        name: String,
        category: &'static str,
        node_count: usize,
        stop: String,
        /// Raw `classes.len()` — allocations, which `union` never reclaims.
        classes_raw: usize,
        /// Canonical classes only (`find(i) == i`).
        live_classes: usize,
        /// E-nodes reachable through `find`, i.e. summed over canonical
        /// classes only. THIS is what orphaning subtracts from: a node
        /// written back to a merged-away slot still occupies memory but is
        /// invisible to `nodes()`, to matching, and to extraction.
        reachable_nodes: usize,
        /// E-nodes stranded in non-canonical slots. Zero is the invariant
        /// the write-back fix restores; every one of these is an extraction
        /// alternative the graph proved and then lost.
        orphaned_nodes: usize,
        /// `arena_static_cost` under `CostModel::latency_prior()` of the
        /// extraction production would emit.
        cost: usize,
        /// Node count of that same extraction — the emitted kernel's size.
        extracted_nodes: usize,
    }

    /// Count e-nodes sitting in slots `find` no longer routes to.
    ///
    /// `union` empties a merged-away class with `mem::take`, so in a healthy
    /// graph every non-canonical slot is empty and this is 0. A non-zero
    /// count means something wrote to a class after it stopped being
    /// canonical — which is exactly the `rebuild_budgeted` write-back bug.
    fn orphaned_node_count(egraph: &EGraph) -> usize {
        (0..egraph.classes.len())
            .filter(|&i| egraph.find(EClassId(i as u32)) != EClassId(i as u32))
            .map(|i| egraph.classes[i].nodes.len())
            .sum()
    }

    /// Sum of e-nodes over canonical classes — everything still reachable
    /// through the public `nodes()`/`tags()` API.
    fn reachable_node_count(egraph: &EGraph) -> usize {
        (0..egraph.classes.len())
            .filter(|&i| egraph.find(EClassId(i as u32)) == EClassId(i as u32))
            .map(|i| egraph.classes[i].nodes.len())
            .sum()
    }

    /// Blast radius of the `rebuild_budgeted` write-back fix: for every
    /// kernel in the real-kernel corpus, the production regime's extracted
    /// cost under `CostModel::latency_prior()` plus the graph-shape numbers
    /// that explain any change.
    ///
    /// **The measurement is a diff of two runs of this same test** — one on
    /// a tree whose write-back targets `self.find(id)` (fixed) and one on a
    /// tree whose write-back targets `id` (the orphaning bug). The two
    /// behaviours deliberately cannot coexist in one binary: a runtime
    /// switch would mean keeping the bug alive in production code to
    /// measure it. Procedure and result:
    /// `docs/results/2026-09-02-rebuild-writeback-orphan.md`.
    ///
    /// `orphaned_nodes` is the direct observable and needs no diff at all —
    /// it is 0 for every kernel iff the write-back is correct.
    #[test]
    #[ignore = "offline measurement: PIXELFLOW_CONGRUENCE_ARENA_DIR=<dir of .arena dumps> PIXELFLOW_REPAIR_WRITEBACK_OUT=<csv> cargo test -p pixelflow-search --release --lib -- --ignored repair_writeback_blast_radius --nocapture"]
    fn repair_writeback_blast_radius() {
        let out = PathBuf::from(
            std::env::var("PIXELFLOW_REPAIR_WRITEBACK_OUT")
                .expect("PIXELFLOW_REPAIR_WRITEBACK_OUT must be set"),
        );
        let paths = arena_corpus_paths();

        let mut rows: Vec<RepairRow> = Vec::new();
        for path in &paths {
            let (name, arena, root) = load_arena_dump(path);
            let category = category_of(&path.file_name().unwrap().to_string_lossy());
            let run = run_production(&name, RuleOrder::Production, &arena, root);
            rows.push(RepairRow {
                name,
                category,
                node_count: run.node_count,
                stop: format!("{:?}", run.stop),
                classes_raw: run.egraph.classes.len(),
                live_classes: live_class_count(&run.egraph),
                reachable_nodes: reachable_node_count(&run.egraph),
                orphaned_nodes: orphaned_node_count(&run.egraph),
                cost: run.cost,
                extracted_nodes: run.extracted_nodes,
            });
        }

        let total_orphaned: usize = rows.iter().map(|r| r.orphaned_nodes).sum();
        let kernels_with_orphans = rows.iter().filter(|r| r.orphaned_nodes > 0).count();
        let total_reachable: usize = rows.iter().map(|r| r.reachable_nodes).sum();
        let total_cost: usize = rows.iter().map(|r| r.cost).sum();
        eprintln!(
            "repair write-back probe: {} kernels, {total_orphaned} orphaned e-nodes \
             across {kernels_with_orphans} kernels ({:.4}% of {total_reachable} reachable), \
             pooled extracted cost {total_cost}",
            rows.len(),
            if total_reachable > 0 {
                total_orphaned as f64 / total_reachable as f64 * 100.0
            } else {
                0.0
            },
        );

        use std::fmt::Write as _;
        let mut csv = String::new();
        writeln!(
            csv,
            "name,category,node_count,stop,classes_raw,live_classes,\
             reachable_nodes,orphaned_nodes,cost,extracted_nodes"
        )
        .unwrap();
        for r in &rows {
            writeln!(
                csv,
                "{},{},{},{},{},{},{},{},{},{}",
                csv_escape(&r.name),
                r.category,
                r.node_count,
                r.stop,
                r.classes_raw,
                r.live_classes,
                r.reachable_nodes,
                r.orphaned_nodes,
                r.cost,
                r.extracted_nodes,
            )
            .unwrap();
        }
        std::fs::write(&out, csv).unwrap_or_else(|e| panic!("write {}: {e}", out.display()));
        eprintln!("wrote {}", out.display());
    }

    /// The invariant the write-back fix restores, asserted on the real
    /// corpus rather than on a hand-built graph: after production
    /// saturation, no e-node may sit in a class `find` no longer routes to.
    /// `union` empties a merged-away class with `mem::take`, so the only way
    /// to strand one is to write to a slot after it stopped being canonical.
    ///
    /// This is the corpus-scale complement to
    /// `rebuild_budgeted_does_not_orphan_nodes_when_current_class_is_merged_away`:
    /// that test proves the mechanism, this one proves it does not happen
    /// anywhere in the kernels production actually compiles.
    #[test]
    #[ignore = "corpus invariant, needs the arena dumps: PIXELFLOW_CONGRUENCE_ARENA_DIR=<dir of .arena dumps> cargo test -p pixelflow-search --release --lib -- --ignored saturation_strands_no_enodes"]
    fn saturation_strands_no_enodes() {
        for path in &arena_corpus_paths() {
            let (name, arena, root) = load_arena_dump(path);
            let run = run_production(&name, RuleOrder::Production, &arena, root);
            assert_eq!(
                orphaned_node_count(&run.egraph),
                0,
                "{name}: production saturation stranded e-nodes in merged-away \
                 classes — rebuild_budgeted wrote back to `id` instead of `find(id)`"
            );
        }
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
}

#[cfg(test)]
pub(crate) mod production_telemetry {
    use super::*;
    use crate::egraph::{Budget, CostModel, Optimizer, SaturationConfig, SaturationStop};
    use pixelflow_ir::arena::{BufferDecl, BufferIdentity};
    use std::fmt::Write as _;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    const DIR_VAR: &str = "PIXELFLOW_TELEMETRY_DIR";
    const OUT_VAR: &str = "PIXELFLOW_TELEMETRY_OUT";
    const REF_MULT_VAR: &str = "PIXELFLOW_TELEMETRY_REF_MULT";
    const KERNEL_CEILING_VAR: &str = "PIXELFLOW_TELEMETRY_KERNEL_CEILING_S";
    /// Generous-run multiplier over the production tier's iteration cap
    /// (and, for the cap-lifted run, its class cap), mirroring the Guide
    /// registration's "unguided-at-4B" comparison.
    const DEFAULT_REF_MULT: usize = 4;
    /// Per-KERNEL wall-clock ceiling shared by the two generous runs. It is
    /// a harness bound, not a metric: a generous run it cuts reports
    /// `Timeout` from the loop, the row's loss against that run is `NA`,
    /// and the row is listed under "loss unmeasured" — never skipped.
    const DEFAULT_KERNEL_CEILING_S: u64 = 1200;

    fn env_required(var: &str) -> String {
        std::env::var(var).unwrap_or_else(|e| panic!("{var} must be set ({e})"))
    }

    /// Inverse of the dumpers' `dump_arena` (cell_grid.rs tests /
    /// pixelflow-graphics/tests/production_glyph_arena_dump.rs): replays
    /// reachable nodes in original id order through the public `push_*`
    /// API, which never hash-conses, so the rebuilt arena has exactly the
    /// dumped node multiset and `reachable_count` is preserved. Buffer
    /// identities are re-minted per distinct dumped ordinal, preserving the
    /// equality structure (two slots naming one buffer still share an
    /// identity) without depending on the dumping process's counter.
    pub(crate) fn load_arena(path: &Path) -> (String, ExprArena, ExprId) {
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
                ["B", slot] => arena.push_buffer(pixelflow_ir::arena::BufferId(
                    slot.parse().expect("buffer slot"),
                )),
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

    /// Latency-prior cost of the arena the JIT would actually execute: the
    /// per-op table summed over every reachable operation once (DAG cost;
    /// leaves are free, as in `CostModel::node_op_cost`). This is the
    /// quality metric — NOT `ExtractedDAG::total_cost`, which is a TREE cost
    /// and pays a shared subterm once per use. `ExtractedDAG::dag_cost`
    /// (#1111) is this same number read off the choices instead of the
    /// materialized arena; this walk stays as the independent check.
    fn arena_cost(arena: &ExprArena, root: ExprId, costs: &CostModel) -> usize {
        let len = arena.nodes_raw().len();
        let mut seen = vec![false; len];
        let mut stack = vec![root];
        let mut total = 0usize;
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            let kind = match arena.node(id) {
                ExprNode::Var(_)
                | ExprNode::Const(_)
                | ExprNode::Buffer(_)
                | ExprNode::Uniform(_) => None,
                ExprNode::Unary(k, _)
                | ExprNode::Binary(k, _, _)
                | ExprNode::Ternary(k, _, _, _) => Some(*k),
                // A fold survives extraction now; the legalizer unrolls it
                // afterwards, and this walk prices the node it is.
                ExprNode::Reduce { .. } => Some(OpKind::Reduce),
                other @ (ExprNode::Param(_)
                | ExprNode::Nary(..)
                | ExprNode::Ref(_)
                | ExprNode::Guard { .. }) => {
                    panic!("extracted arena contains {other:?}")
                }
            };
            if let Some(k) = kind {
                assert_ne!(k, OpKind::Dwrt, "Dwrt survived extraction");
                total += costs.cost(k);
            }
            stack.extend(arena.children(id));
        }
        total
    }

    struct Run {
        stop: SaturationStop,
        iterations: usize,
        total_unions: usize,
        classes_after: usize,
        applications: usize,
        journal_unions: usize,
        elapsed: Duration,
        cost: usize,
        dp_cost: usize,
        extracted_nodes: usize,
    }

    impl Run {
        /// The budget-independent trajectory signature: two runs of the same
        /// arena that were never cut differently agree on all of these.
        fn signature(&self) -> (usize, usize, usize, usize, usize) {
            (
                self.iterations,
                self.total_unions,
                self.classes_after,
                self.applications,
                self.cost,
            )
        }
    }

    /// The production sequence of `optimize_runtime_arena_uncached`
    /// (`runtime.rs:106-137`) from the e-graph build onward, with the budget
    /// as parameters so the same function runs the production tier and both
    /// generous runs. `lower_dwrt_owned` (`:120`) and `config_for_node_count`
    /// (`:126-127`) run once in the caller since they are budget-independent.
    fn run(
        arena: &ExprArena,
        root: ExprId,
        max_iterations: usize,
        max_classes: usize,
        timeout: Duration,
    ) -> Run {
        // Driven through `Optimizer` — the one entry point production itself
        // uses since #1108 — with `Budget::Explicit` so the caps stay
        // parameters, which this measurement varies between its production
        // and generous regimes.
        //
        // The pre-#1108 version asserted `ExtractionPolicy::Static` here, to
        // stop a stray `PIXELFLOW_NNUE_WEIGHTS` from silently changing what
        // was being measured. `env_extraction_policy` no longer exists and
        // production has no env-driven policy path, so there is nothing left
        // to guard: `Optimizer::production()` IS the static latency prior.
        let mut optimizer = Optimizer::production()
            .budget(Budget::Explicit {
                iterations: max_iterations,
                classes: max_classes,
                applications: None,
            })
            .hard_ceiling(timeout);

        let mut egraph = optimizer.egraph();
        let root_class =
            crate::egraph::insert(arena, root, &mut egraph, crate::egraph::Vocabulary::Runtime)
                .expect("production arena must be e-graph representable (no Param/Nary)");

        let started = Instant::now();
        let optimized = optimizer.run(
            &mut egraph,
            root_class,
            crate::egraph::reachable_count(arena, root),
        );
        let elapsed = started.elapsed();

        let (extracted, extracted_root) = optimized.to_arena(&egraph, root_class);

        let costs = CostModel::latency_prior();
        // The DP's own objective value for the term it returned: a TREE cost,
        // so sharing is not priced and the number is not comparable with
        // `arena_cost` below (which is the DAG cost the kernel pays). Since
        // #1111 the optimizer reports it directly from the settled choices,
        // so this column no longer needs a second extraction to recover it —
        // and no longer carries the pre-repair/`CYCLE_COST` inflation that
        // made it describe a term other than the one extracted.
        let dp_cost = optimized.cost.tree;

        Run {
            stop: optimized.stats.stop,
            iterations: optimized.stats.iterations,
            total_unions: optimized.stats.unions,
            classes_after: optimized.stats.classes,
            applications: optimized.stats.applications as usize,
            journal_unions: egraph.provenance().union_count(),
            elapsed,
            cost: arena_cost(&extracted, extracted_root, &costs),
            dp_cost,
            extracted_nodes: crate::egraph::reachable_count(&extracted, extracted_root),
        }
    }

    fn tier_name(config: &SaturationConfig) -> &'static str {
        match config.max_iterations {
            20 => "blitz",
            50 => "rapid",
            100 => "classical",
            other => panic!("unknown tier with max_iterations={other}"),
        }
    }

    /// Production's cost over a generous run's, as a percentage. `None` when
    /// the generous run was cut by the harness ceiling (nothing to compare
    /// against) or when both costs are zero (the 1-node space glyph: 0/0 is
    /// undefined, not 0%).
    fn loss_pct(prod: &Run, generous: &Run) -> Option<f64> {
        if generous.stop == SaturationStop::Timeout {
            return None;
        }
        if generous.cost == 0 {
            assert_eq!(
                prod.cost, 0,
                "generous run extracted cost 0 but production did not"
            );
            return None;
        }
        Some((prod.cost as f64 - generous.cost as f64) / generous.cost as f64 * 100.0)
    }

    fn fmt_opt(v: Option<f64>) -> String {
        v.map_or_else(|| "NA".to_string(), |v| format!("{v:.2}"))
    }

    fn median(v: &[f64]) -> f64 {
        assert!(!v.is_empty(), "median of nothing");
        let mut v = v.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let n = v.len();
        if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        }
    }

    fn percentile(v: &[f64], p: f64) -> f64 {
        assert!(!v.is_empty(), "percentile of nothing");
        let mut v = v.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let idx = ((v.len() - 1) as f64 * p).round() as usize;
        v[idx]
    }

    /// `uptime`'s load averages, recorded so a run can be labelled quiet or
    /// loaded. The 200ms production wall clock is the one budget whose
    /// verdict depends on the host, so this is part of the measurement.
    fn load_averages() -> String {
        let out = std::process::Command::new("uptime")
            .output()
            .expect("uptime must be runnable to record host load");
        assert!(out.status.success(), "uptime failed: {out:?}");
        let text = String::from_utf8(out.stdout).expect("uptime output is utf-8");
        let idx = text
            .find("load average")
            .unwrap_or_else(|| panic!("uptime output has no load average: {text:?}"));
        text[idx..].trim().to_string()
    }

    #[test]
    #[ignore = "measurement: PIXELFLOW_TELEMETRY_DIR=<dumps> PIXELFLOW_TELEMETRY_OUT=<tsv> cargo test -p pixelflow-search --release -- --ignored production_saturation_telemetry --nocapture --test-threads=1"]
    fn production_saturation_telemetry() {
        let dir = PathBuf::from(env_required(DIR_VAR));
        let out_path = PathBuf::from(env_required(OUT_VAR));
        let mult: usize = std::env::var(REF_MULT_VAR)
            .map(|s| s.parse().expect("REF_MULT must be an integer"))
            .unwrap_or(DEFAULT_REF_MULT);
        let ceiling = Duration::from_secs(
            std::env::var(KERNEL_CEILING_VAR)
                .map(|s| s.parse().expect("KERNEL_CEILING_S must be an integer"))
                .unwrap_or(DEFAULT_KERNEL_CEILING_S),
        );
        assert!(
            std::env::var("PIXELFLOW_NNUE_WEIGHTS").is_err(),
            "PIXELFLOW_NNUE_WEIGHTS is set; this measures the default production policy — unset it"
        );

        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
            .map(|e| e.expect("dir entry").path())
            .filter(|p| p.extension().is_some_and(|x| x == "arena"))
            .collect();
        files.sort();
        assert!(!files.is_empty(), "no *.arena files in {}", dir.display());

        let load_start = load_averages();
        println!("host load at start: {load_start}");

        let header = "name\tgroup\tnodes\ttier\tstop\tmachine_dependent\titers\tmax_iters\tclasses\tmax_classes\tapps\tunions\tjournal_unions\telapsed_ms\tcost\tdp_cost\text_nodes\
                      \tref_stop\tref_iters\tref_classes\tref_apps\tref_elapsed_ms\tref_cost\tloss_vs_ref_pct\
                      \tlifted_stop\tlifted_iters\tlifted_classes\tlifted_apps\tlifted_elapsed_ms\tlifted_cost\tloss_vs_lifted_pct\tcap_lift_changed\tanomaly";
        let mut tsv = String::new();
        writeln!(tsv, "{header}").expect("write");
        println!("{header}");

        struct Row {
            name: String,
            group: String,
            stop: SaturationStop,
            ref_stop: SaturationStop,
            lifted_stop: SaturationStop,
            loss_vs_ref: Option<f64>,
            loss_vs_lifted: Option<f64>,
            apps: usize,
            anomaly: Option<String>,
            fatal: bool,
        }
        let mut rows: Vec<Row> = Vec::new();

        for path in &files {
            let (name, raw_arena, raw_root) = load_arena(path);
            let group = name.split(':').next().expect("group prefix").to_string();

            // runtime.rs:120
            let (arena, root) = pixelflow_ir::passes::lower_dwrt_owned(&raw_arena, raw_root)
                .unwrap_or_else(|e| panic!("{name}: lower_dwrt failed: {e:?}"));
            // runtime.rs:126-127
            let node_count = crate::egraph::reachable_count(&arena, root);
            let config = crate::egraph::saturate::config_for_node_count(node_count);

            let prod = run(
                &arena,
                root,
                config.max_iterations,
                config.max_classes,
                config.safety_ceiling,
            );

            // Two generous runs share one per-kernel ceiling: `refr` keeps
            // production's class cap and lifts only the iterations and the
            // clock (so its loss is the clock's bite alone); `lifted` lifts
            // the class cap too (the whole budget's bite).
            let ref_iters = config.max_iterations * mult;
            let lifted_classes = config.max_classes * mult;
            let refr = run(&arena, root, ref_iters, config.max_classes, ceiling);
            let remaining = ceiling.saturating_sub(refr.elapsed);
            let lifted = run(&arena, root, ref_iters, lifted_classes, remaining);

            let loss_vs_ref = loss_pct(&prod, &refr);
            let loss_vs_lifted = loss_pct(&prod, &lifted);
            let cap_lift_changed = refr.signature() != lifted.signature();

            let mut anomaly: Vec<String> = Vec::new();
            let mut fatal = false;
            let clock_did_not_decide = matches!(
                prod.stop,
                SaturationStop::Quiesced | SaturationStop::ClassCap
            );
            if clock_did_not_decide
                && refr.stop != SaturationStop::Timeout
                && prod.signature() != refr.signature()
            {
                // Production stopped on its own (no clock involved), so the
                // same-cap run with more iterations and no clock must retrace
                // it exactly. If it does not, the optimizer is
                // nondeterministic and the row is not trustworthy.
                fatal = true;
                anomaly.push(format!(
                    "production stopped {:?} but the same-cap generous run diverged: prod(iters={},unions={},classes={},apps={},cost={}) ref(iters={},unions={},classes={},apps={},cost={})",
                    prod.stop,
                    prod.iterations,
                    prod.total_unions,
                    prod.classes_after,
                    prod.applications,
                    prod.cost,
                    refr.iterations,
                    refr.total_unions,
                    refr.classes_after,
                    refr.applications,
                    refr.cost
                ));
            }
            if refr.stop == SaturationStop::Timeout {
                anomaly.push(format!(
                    "same-cap generous run cut by the {ceiling:?} harness ceiling after {} iterations: loss_vs_ref unmeasured",
                    refr.iterations
                ));
            }
            if lifted.stop == SaturationStop::Timeout {
                anomaly.push(format!(
                    "cap-lifted generous run cut by the harness ceiling ({remaining:?} left of {ceiling:?}) after {} iterations: loss_vs_lifted unmeasured",
                    lifted.iterations
                ));
            }
            if loss_vs_ref.is_some_and(|l| l < 0.0)
                || loss_vs_lifted.is_some_and(|l| l < 0.0)
                || (lifted.stop != SaturationStop::Timeout
                    && refr.stop != SaturationStop::Timeout
                    && refr.cost < lifted.cost)
            {
                anomaly.push(format!(
                    "more saturation extracted WORSE: cost prod={} ref={} lifted={}",
                    prod.cost, refr.cost, lifted.cost
                ));
            }
            let anomaly = (!anomaly.is_empty()).then(|| anomaly.join("; "));

            let line = format!(
                "{name}\t{group}\t{node_count}\t{}\t{:?}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.1}\t{}\t{}\t{}\
                 \t{:?}\t{}\t{}\t{}\t{:.1}\t{}\t{}\
                 \t{:?}\t{}\t{}\t{}\t{:.1}\t{}\t{}\t{}\t{}",
                tier_name(&config),
                prod.stop,
                prod.stop == SaturationStop::Timeout,
                prod.iterations,
                config.max_iterations,
                prod.classes_after,
                config.max_classes,
                prod.applications,
                prod.total_unions,
                prod.journal_unions,
                prod.elapsed.as_secs_f64() * 1e3,
                prod.cost,
                prod.dp_cost,
                prod.extracted_nodes,
                refr.stop,
                refr.iterations,
                refr.classes_after,
                refr.applications,
                refr.elapsed.as_secs_f64() * 1e3,
                refr.cost,
                fmt_opt(loss_vs_ref),
                lifted.stop,
                lifted.iterations,
                lifted.classes_after,
                lifted.applications,
                lifted.elapsed.as_secs_f64() * 1e3,
                lifted.cost,
                fmt_opt(loss_vs_lifted),
                cap_lift_changed,
                anomaly.as_deref().unwrap_or("-"),
            );
            println!("{line}");
            writeln!(tsv, "{line}").expect("write");
            rows.push(Row {
                name: name.clone(),
                group,
                stop: prod.stop,
                ref_stop: refr.stop,
                lifted_stop: lifted.stop,
                loss_vs_ref,
                loss_vs_lifted,
                apps: prod.applications,
                anomaly,
                fatal,
            });
        }

        assert_eq!(
            rows.len(),
            files.len(),
            "every dumped kernel must produce a row"
        );
        std::fs::write(&out_path, &tsv)
            .unwrap_or_else(|e| panic!("write {}: {e}", out_path.display()));
        let load_end = load_averages();
        let meta_path = out_path.with_extension("meta");
        std::fs::write(
            &meta_path,
            format!(
                "kernels\t{}\nref_mult\t{mult}\nkernel_ceiling_s\t{}\nload_start\t{load_start}\nload_end\t{load_end}\n",
                rows.len(),
                ceiling.as_secs()
            ),
        )
        .unwrap_or_else(|e| panic!("write {}: {e}", meta_path.display()));
        println!("host load at end: {load_end}");

        let mut groups: Vec<String> = rows.iter().map(|r| r.group.clone()).collect();
        groups.sort();
        groups.dedup();
        groups.push("ALL".to_string());
        println!(
            "\n== summary (stop = SaturationResult::stop; ref = {mult}x iterations/same cap; lifted = {mult}x iterations/{mult}x cap; both without production's clock, under a {ceiling:?} per-kernel ceiling) =="
        );
        println!(
            "group\tn\tquiesced\titer_ceiling\tclass_cap\ttimeout\tref_class_cap\tlifted_class_cap\tn_loss_ref\tmed_loss_ref%\tp90_loss_ref%\tmax_loss_ref%\tn_loss_lifted\tmed_loss_lifted%\tp90_loss_lifted%\tmax_loss_lifted%\tmed_apps\tmax_apps"
        );
        for g in &groups {
            let sel: Vec<&Row> = rows
                .iter()
                .filter(|r| g == "ALL" || &r.group == g)
                .collect();
            let count = |s: SaturationStop| sel.iter().filter(|r| r.stop == s).count();
            let lr: Vec<f64> = sel.iter().filter_map(|r| r.loss_vs_ref).collect();
            let ll: Vec<f64> = sel.iter().filter_map(|r| r.loss_vs_lifted).collect();
            let apps: Vec<f64> = sel.iter().map(|r| r.apps as f64).collect();
            let q = |v: &[f64]| {
                if v.is_empty() {
                    "NA\tNA\tNA".to_string()
                } else {
                    format!(
                        "{:.2}\t{:.2}\t{:.2}",
                        median(v),
                        percentile(v, 0.9),
                        percentile(v, 1.0)
                    )
                }
            };
            println!(
                "{g}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.0}\t{:.0}",
                sel.len(),
                count(SaturationStop::Quiesced),
                count(SaturationStop::IterationCeiling),
                count(SaturationStop::ClassCap),
                count(SaturationStop::Timeout),
                sel.iter()
                    .filter(|r| r.ref_stop == SaturationStop::ClassCap)
                    .count(),
                sel.iter()
                    .filter(|r| r.lifted_stop == SaturationStop::ClassCap)
                    .count(),
                lr.len(),
                q(&lr),
                ll.len(),
                q(&ll),
                median(&apps),
                percentile(&apps, 1.0),
            );
        }
        let worse: Vec<&Row> = rows
            .iter()
            .filter(|r| r.anomaly.as_deref().is_some_and(|a| a.contains("WORSE")))
            .collect();
        println!(
            "\nnon-fatal anomalies (more saturation extracted worse): {}",
            worse.len()
        );
        let ceiling_hits: Vec<&Row> = rows
            .iter()
            .filter(|r| {
                r.ref_stop == SaturationStop::Timeout || r.lifted_stop == SaturationStop::Timeout
            })
            .collect();
        println!(
            "loss unmeasured (a generous run hit the {ceiling:?} per-kernel harness ceiling): {}{}",
            ceiling_hits.len(),
            if ceiling_hits.is_empty() {
                String::new()
            } else {
                format!(
                    " — {}",
                    ceiling_hits
                        .iter()
                        .map(|r| r.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        );

        let fatal: Vec<String> = rows
            .iter()
            .filter(|r| r.fatal)
            .map(|r| format!("{}: {}", r.name, r.anomaly.as_deref().unwrap_or("?")))
            .collect();
        assert!(
            fatal.is_empty(),
            "table complete ({} rows, written to {}) but {} row(s) are not trustworthy:\n{}",
            rows.len(),
            out_path.display(),
            fatal.len(),
            fatal.join("\n")
        );
    }
}

/// The neutrality check every Phase 3 optimizer-lever PR owes: replay the
/// arenas production actually compiles through the production entry point,
/// and digest what comes out.
///
/// The research levers ([`Optimizer::guide`](crate::egraph::Optimizer::guide),
/// [`Optimizer::rerank`](crate::egraph::Optimizer::rerank),
/// [`Optimizer::mask`](crate::egraph::Optimizer::mask)) are all
/// `Option`s that [`Optimizer::production`](crate::egraph::Optimizer::production)
/// leaves `None`, so adding one may not move a single production byte. L4
/// (`docs/plans/2026-09-02-optimizer-api.md`) says a lever cannot change
/// *meaning*; this says the stronger thing a port needs — that it did not
/// change the *term*, either.
///
/// Run it on both sides of a change and diff the two TSVs:
///
/// ```text
/// PIXELFLOW_EQUIV_DIR=/private/tmp/classcap_corpus \
/// PIXELFLOW_EQUIV_OUT=/tmp/before.tsv \
///   cargo test -p pixelflow-search --release --test-threads=1 \
///     -- --ignored production_extraction_digest --nocapture
/// ```
#[cfg(test)]
mod production_equivalence {
    use super::production_telemetry::load_arena;
    use super::*;
    use std::fmt::Write as _;

    const DIR_VAR: &str = "PIXELFLOW_EQUIV_DIR";
    const OUT_VAR: &str = "PIXELFLOW_EQUIV_OUT";

    /// FNV-1a over the optimized arena's canonical serialization — the same
    /// stable digest [`RuleId`](crate::egraph::RuleId) uses, for the same
    /// reason: it has to be reproducible by a different build.
    fn digest(arena: &ExprArena, root: ExprId) -> String {
        let mut text = String::new();
        let len = arena.nodes_raw().len();
        let mut reachable = vec![false; len];
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut reachable[id.0 as usize], true) {
                continue;
            }
            stack.extend(arena.children(id));
        }
        let mut nodes = 0usize;
        for (idx, live) in reachable.iter().enumerate() {
            if !*live {
                continue;
            }
            nodes += 1;
            writeln!(text, "{idx} {:?}", arena.node(ExprId(idx as u32))).expect("fmt");
        }
        writeln!(text, "root {}", root.0).expect("fmt");
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in text.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        format!("{h:016x}\t{nodes}")
    }

    #[test]
    #[ignore = "equivalence check: PIXELFLOW_EQUIV_DIR=<dumps> PIXELFLOW_EQUIV_OUT=<tsv> cargo test -p pixelflow-search --release -- --ignored production_extraction_digest --nocapture"]
    fn production_extraction_digest() {
        let dir = std::path::PathBuf::from(
            std::env::var(DIR_VAR).unwrap_or_else(|e| panic!("{DIR_VAR} must be set ({e})")),
        );
        let out = std::path::PathBuf::from(
            std::env::var(OUT_VAR).unwrap_or_else(|e| panic!("{OUT_VAR} must be set ({e})")),
        );
        let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
            .map(|e| e.expect("dir entry").path())
            .filter(|p| p.extension().is_some_and(|x| x == "arena"))
            .collect();
        paths.sort();
        assert!(!paths.is_empty(), "{}: no .arena dumps", dir.display());

        let mut text = String::new();
        for path in &paths {
            let (name, arena, root) = load_arena(path);
            // The production entry point itself, uncached — a static cache
            // would make the second kernel with an equal arena report the
            // first one's answer rather than recomputing it.
            let line = match optimize_runtime_arena_uncached(&arena, root, LatticeShape::POINT) {
                Some((opt, opt_root)) => digest(&opt, opt_root),
                // `None` is a real production outcome (an arena
                // `optimize_runtime_arena` bails on), and it must stay the
                // same outcome across the change, so it is a row, not a skip.
                None => String::from("BAILED\t0"),
            };
            writeln!(text, "{name}\t{line}").expect("fmt");
        }
        std::fs::write(&out, &text).unwrap_or_else(|e| panic!("write {}: {e}", out.display()));
        println!(
            "digested {} production arenas -> {}",
            paths.len(),
            out.display()
        );
    }
}
