//! What the [`Ir`] boundary is entitled to, pinned.
//!
//! These are the tests that had nowhere to live before: insertion was two
//! private functions with different failure modes, so "which ops may a
//! macro-tier graph hold?" and "does inserting spend budget on unreachable
//! nodes?" were properties of whichever helper a call site happened to call.
//! Naming the boundary is what makes them assertable.

use pixelflow_ir::ExprBuilder;
use pixelflow_ir::arena::{BufferDecl, BufferIdentity, ExprNode, UniformDecl, UniformIdentity};
use pixelflow_ir::{Children, ExprArena, Ir, Kernel, KernelStore, OpKind, Shape};
use pixelflow_search::egraph::{Declined, EGraph, Vocabulary, insert, reachable_count};
use pixelflow_search::runtime::optimize_runtime_arena;

/// A fresh uniform declaration with the given default.
fn uniform(default: f32) -> UniformDecl {
    UniformDecl {
        id: UniformIdentity::mint(),
        default,
    }
}

/// A uniform inserts as a leaf of its own, and one identity is one e-class
/// however many leaves name it — the same hash-consing a `Buffer` gets — while
/// two instances with equal defaults stay two classes.
#[test]
fn uniform_inserts_and_hash_conses_by_identity() {
    let same = uniform(1.0);
    let other = uniform(1.0);
    let mut arena = ExprArena::new();
    let a = arena.embed(Shape::Uniform(same));
    let b = arena.embed(Shape::Uniform(same));
    let c = arena.embed(Shape::Uniform(other));
    assert_eq!(arena.uniforms().len(), 2, "one slot per identity");
    let ab = arena.push_binary(OpKind::Add, a, b);
    let root = arena.push_binary(OpKind::Add, ab, c);

    let mut eg = EGraph::new();
    let class = insert(&arena, root, &mut eg, Vocabulary::Runtime).expect("uniforms insert");
    let ca = insert(&arena, a, &mut eg, Vocabulary::Runtime).expect("insert");
    let cb = insert(&arena, b, &mut eg, Vocabulary::Runtime).expect("insert");
    let cc = insert(&arena, c, &mut eg, Vocabulary::Runtime).expect("insert");
    assert_eq!(eg.find(ca), eg.find(cb), "one instance is one e-class");
    assert_ne!(eg.find(ca), eg.find(cc), "two instances are two e-classes");
    let _ = class;
    // The macro tier holds it too: a uniform is not a runtime-only op.
    let mut eg = EGraph::new();
    assert!(insert(&arena, root, &mut eg, Vocabulary::Templates).is_ok());
}

#[test]
fn graph_insert_uses_the_opaque_dag_surface() {
    let mut builder = ExprBuilder::new();
    let x = builder.var(0);
    let one = builder.constant(1.0);
    let root = builder.binary(OpKind::Add, x, one);
    let graph = builder.finish_one(root);

    let mut egraph = EGraph::new();
    let root_class =
        pixelflow_search::egraph::insert_graph(&graph, &mut egraph, Vocabulary::Runtime)
            .expect("graph with supported nodes must insert");
    assert_eq!(egraph.num_classes(), 3);
    assert_eq!(pixelflow_search::egraph::reachable_count_graph(&graph), 3);
    assert!(
        egraph
            .nodes(root_class)
            .iter()
            .any(|node| node.op().is_some())
    );
}

/// The macro tier must not hold mask or integer-domain ops.
///
/// This is a soundness rule, not a preference: that e-graph runs at
/// macro-expansion time, BEFORE composition, where resolving the `Dwrt` nodes
/// these ops travel with is wrong (a leaf's `DX` is 1 only until an enclosing
/// `.at()` warp scales it — the fonts' density-dependent AA ramp broke exactly
/// this way when these ops were briefly global). It lived only in a comment
/// until [`Vocabulary`] gave it a name.
#[test]
fn templates_vocabulary_refuses_the_runtime_only_ops() {
    for kind in [
        OpKind::BitAnd,
        OpKind::BitOr,
        OpKind::TruncToInt,
        OpKind::IntToFloat,
        OpKind::IAdd,
        OpKind::Shl,
        OpKind::Shr,
    ] {
        assert!(
            Vocabulary::Templates.resolve(kind).is_none(),
            "{kind:?} must not be holdable by a macro-tier e-graph — it is \
             unsound before composition"
        );
        assert!(
            Vocabulary::Runtime.resolve(kind).is_some(),
            "{kind:?} must be holdable by a runtime-tier e-graph, or the \
             production frame kernel loses CSE across its channels"
        );
    }
}

/// `Gather` is holdable in both tiers but nameable by no rewrite template —
/// its whole participation is hash-consing CSE.
#[test]
fn gather_is_holdable_everywhere_and_nameable_nowhere() {
    assert!(Vocabulary::Templates.resolve(OpKind::Gather).is_some());
    assert!(Vocabulary::Runtime.resolve(OpKind::Gather).is_some());
    assert!(
        pixelflow_search::egraph::ops::op_from_kind(OpKind::Gather).is_none(),
        "a rewrite template must not be able to name Gather"
    );
}

/// Insertion walks what is reachable from the root, not what the arena
/// happens to hold.
///
/// An append-only arena rewritten in place carries abandoned nodes. Inserting
/// those spends `max_classes` — a budget dimension — on an expression nobody
/// asked about, making saturation a function of allocation history. Callers
/// used to compact first to avoid it; now they do not have to.
#[test]
fn insertion_ignores_unreachable_nodes() {
    let mut arena = ExprArena::new();
    let x = arena.push_var(0);
    let y = arena.push_var(1);
    let root = arena.push_binary(OpKind::Add, x, y);

    // Abandoned: built, then never referenced from `root`.
    let g = arena.push_var(2);
    let _garbage = arena.push_binary(OpKind::Mul, g, g);

    assert_eq!(arena.len(), 5, "arena holds the garbage");
    assert_eq!(reachable_count(&arena, root), 3, "but only 3 are reachable");

    let mut eg = EGraph::new();
    let _ = insert(&arena, root, &mut eg, Vocabulary::Templates).expect("insert");
    assert_eq!(
        eg.num_classes(),
        3,
        "the e-graph must hold only the reachable expression"
    );
}

/// A term the vocabulary cannot express is declined, never a panic: the caller
/// compiles it unoptimized, which is always available because optimization is
/// never required for correctness.
#[test]
fn unrepresentable_input_declines_rather_than_panicking() {
    let mut arena = ExprArena::new();
    let x = arena.push_var(0);
    let masked = arena.push_binary(OpKind::BitAnd, x, x);
    let mut eg = EGraph::new();
    assert_eq!(
        insert(&arena, masked, &mut eg, Vocabulary::Templates),
        Err(Declined::Op(OpKind::BitAnd))
    );
}

/// A `Param` is holdable or not *according to the vocabulary*, which is the
/// question `Vocabulary` exists to answer.
///
/// `Templates` is the macro tier, where a builder's scalar is legitimately
/// unbound — an unbound slot is what a builder is — so it inserts as the
/// opaque leaf `ENode::Param`, alongside `Buffer` and `Uniform`. `Runtime` is
/// bake time, by which a builder must have substituted it, so one surviving
/// there means the term was never specialized and is declined.
///
/// Both halves are asserted here because the interesting property is the
/// *difference*: `insert` used to decline `Param` under every vocabulary, and
/// each macro-side caller then smuggled params past it as something else — as
/// `Var(16 + i)`, as an opaque synthetic identifier. A test that only checked
/// the `Runtime` half would pass just as well against that blanket refusal.
#[test]
fn a_param_is_held_by_the_macro_vocabulary_and_declined_by_the_runtime_one() {
    let mut arena = ExprArena::new();
    let p = arena.push_param(3);

    let mut templates = EGraph::new();
    assert!(
        insert(&arena, p, &mut templates, Vocabulary::Templates).is_ok(),
        "the macro tier must hold an unbound builder slot"
    );

    let mut runtime = EGraph::new();
    assert_eq!(
        insert(&arena, p, &mut runtime, Vocabulary::Runtime),
        Err(Declined::Param(3)),
        "a Param at bake time means a builder was never called"
    );
}

/// A reference is declined, not mishandled. `passes::expand_refs` runs before
/// saturation in every pipeline, so one arriving here is a pipeline-order bug
/// — and the e-graph must say so rather than insert a leaf it cannot rewrite,
/// which would silently make an inlining rule look like it had nothing to do.
#[test]
fn a_reference_is_declined_by_every_vocabulary() {
    let named = Kernel::x().mul(&Kernel::constant(3.0)).by_ref();
    let (arena, root) = named.rooted().entry().marshal(named.environment());
    let key = KernelStore::intern(&Kernel::x().mul(&Kernel::constant(3.0)));

    for vocab in [Vocabulary::Runtime, Vocabulary::Templates] {
        let mut eg = EGraph::new();
        assert_eq!(
            insert(&arena, root, &mut eg, vocab),
            Err(Declined::Ref(key)),
            "{vocab:?} must decline a reference"
        );
    }
}

/// And the runtime tier as a whole does not decline it: `ExpandRefs` runs
/// first, so what reaches the e-graph is the referent's body and the kernel
/// optimizes exactly as the spliced composition does.
#[test]
fn the_runtime_pipeline_expands_before_it_saturates() {
    let body = Kernel::x().mul(&Kernel::constant(0.0)).add(&Kernel::y());
    let named = body.by_ref();
    let (arena, root) = named.rooted().entry().marshal(named.environment());
    let optimized = optimize_runtime_arena(&arena, root, pixelflow_ir::LatticeShape::POINT)
        .expect("a named kernel must optimize, not bail");
    let (opt, opt_root) = &*optimized;
    // X·0 + Y folds to Y, which it could not do without seeing the body.
    assert!(
        matches!(opt.node(*opt_root), ExprNode::Var(1)),
        "expected the referent's body to be folded to bare Y, got {:?}",
        opt.node(*opt_root)
    );
}

/// L3a, materialisation: `embed` after `project` reconstructs the same term.
///
/// The optimizer-api plan recorded this law as UNREPRESENTABLE AS STATED
/// (docs/plans/2026-09-02-optimizer-api.md:188) because there was no boundary
/// to state it against. There is one now, and it is generic over any `Ir`.
#[test]
fn project_then_embed_round_trips() {
    let mut src = ExprArena::new();
    let x = src.push_var(0);
    let y = src.push_var(1);
    let c = src.push_const(2.5);
    let shared = src.push_binary(OpKind::Mul, x, y);
    // `shared` used twice: the round trip must preserve sharing, not expand it.
    let sum = src.push_binary(OpKind::Add, shared, shared);
    let root = src.push_ternary(OpKind::MulAdd, sum, c, x);

    let mut dst = ExprArena::new();
    let dst_root = src.rebuild_into(root, &mut dst);

    assert_eq!(
        dst.len(),
        reachable_count(&src, root),
        "sharing must survive the round trip — a re-expanded DAG would be larger"
    );
    // Same term: both sides insert into an e-graph as the same single class.
    let mut eg = EGraph::new();
    let a = insert(&src, root, &mut eg, Vocabulary::Templates).expect("src");
    let b = insert(&dst, dst_root, &mut eg, Vocabulary::Templates).expect("dst");
    assert_eq!(
        eg.find(a),
        eg.find(b),
        "the rebuilt term must hash-cons into the original's e-class"
    );
}

/// `embed` owns buffer declaration: one slot per distinct identity, however
/// many leaves name it.
#[test]
fn embed_declares_one_slot_per_buffer_identity() {
    let decl = BufferDecl {
        id: BufferIdentity::mint(),
        width: 8,
        height: 4,
    };
    let mut arena = ExprArena::new();
    let a = arena.embed(Shape::Buffer(decl));
    let b = arena.embed(Shape::Buffer(decl));
    assert_eq!(
        arena.buffers().len(),
        1,
        "one identity must claim exactly one slot"
    );
    assert_ne!(a, b, "each leaf is its own node, sharing one slot");
}

/// Projection reports the arity the node actually has, so a caller rebuilding
/// through `embed` cannot silently drop an operand.
#[test]
fn projection_reports_arity() {
    let mut arena = ExprArena::new();
    let x = arena.push_var(0);
    let u = arena.push_unary(OpKind::Sqrt, x);
    let b = arena.push_binary(OpKind::Add, x, u);
    let t = arena.push_ternary(OpKind::MulAdd, x, u, b);

    let arity = |id| match arena.project(id) {
        Shape::Op(_, kids) => kids.len(),
        _ => 0,
    };
    assert_eq!((arity(x), arity(u), arity(b), arity(t)), (0, 1, 2, 3));

    match arena.project(t) {
        Shape::Op(OpKind::MulAdd, kids) => {
            assert_eq!(
                kids.iter().collect::<Vec<_>>(),
                alloc_vec(&[x, u, b]),
                "operands must project in order"
            );
        }
        other => panic!("expected MulAdd, got {other:?}"),
    }

    // Leaves project as leaves, carrying their payload.
    assert!(matches!(arena.project(x), Shape::Var(0)));
    let c = arena.push_const(1.5);
    assert!(matches!(arena.project(c), Shape::Const(v) if v == 1.5));
    let _ = Children::<pixelflow_ir::ExprId>::Zero;
}

fn alloc_vec(ids: &[pixelflow_ir::ExprId]) -> Vec<pixelflow_ir::ExprId> {
    ids.to_vec()
}
