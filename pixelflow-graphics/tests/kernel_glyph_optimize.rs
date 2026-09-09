//! Optimization-quality guards for the glyph coverage kernels.
//!
//! A glyph kernel is the hottest runtime-composed arena in the system (every
//! bake evaluates every reachable node per pixel), and its cost is dominated
//! by the antialiasing ramps: every edge function `d` is normalised by
//! `‖∇d‖ = √(DX(d)² + DY(d)²)`, once per piece per estimate.
//!
//! A glyph is now **two folds over one coefficient table** — a `sum_over`
//! for the winding and a `min_over` for the distance, each with one fixed
//! body reading its numbers by column at its own reduce binder. So the two
//! things worth pinning are that the built arena does not grow with the
//! outline, and that the optimizer's output grows exactly linearly in the
//! piece count with a fixed budget per piece.
//!
//! These tests count surviving operations through the runtime pipeline
//! (`optimize_runtime_arena` → `lower_dwrt`) — the exact stages
//! `Lattice::bake` runs — so a regression in derivative lowering or CSE
//! shows up as a hard number, not a benchmark whisper.

use pixelflow_graphics::fonts::{loop_blinn, Contour, Font, Outline, Segment};
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use pixelflow_ir::passes::{expand_refs_owned, lower_dwrt_owned};
use pixelflow_ir::{Kernel, OpKind};

const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// A glyph kernel's arena with its references linked. The winding sum is
/// composed by reference, and a name has no derivative, declares no buffer,
/// and counts as one node — so every count and every lowering below starts
/// from the linked arena, as the pipeline's own first step does.
fn linked(kernel: &Kernel) -> (ExprArena, ExprId) {
    let (arena, root) = kernel.parts();
    expand_refs_owned(arena, root)
}

/// Count reachable nodes matching `pred` from `root`.
fn count_reachable(arena: &ExprArena, root: ExprId, pred: impl Fn(&ExprNode) -> bool) -> usize {
    let len = arena.nodes_raw().len();
    let mut seen = vec![false; len];
    let mut stack = vec![root];
    let mut count = 0;
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        if pred(arena.node(id)) {
            count += 1;
        }
        stack.extend(arena.children(id));
    }
    count
}

fn count_op(arena: &ExprArena, root: ExprId, op: OpKind) -> usize {
    count_reachable(arena, root, |n| match n {
        ExprNode::Unary(k, _) => *k == op,
        ExprNode::Binary(k, _, _) => *k == op,
        ExprNode::Ternary(k, _, _, _) => *k == op,
        _ => false,
    })
}

fn total_reachable(arena: &ExprArena, root: ExprId) -> usize {
    count_reachable(arena, root, |_| true)
}

/// Run the same optimization stages `Lattice::bake` runs, then lower any
/// residual `Dwrt` exactly as the compile entries do, and report the final
/// (arena, root) the emitter would actually schedule. Prints per-stage
/// counts so a failure localizes to the stage that dropped the ball.
fn bake_pipeline(arena: &ExprArena, root: ExprId, shape: [u32; 2]) -> (ExprArena, ExprId) {
    let optimized = pixelflow_search::runtime::optimize_runtime_arena(
        arena,
        root,
        pixelflow_ir::LatticeShape::new(shape),
    );
    let (a, r) = optimized
        .as_deref()
        .map(|(a, r)| (a.clone(), *r))
        .unwrap_or_else(|| (arena.clone(), root));
    eprintln!(
        "  post-egraph: total={} sqrt={} dwrt={}",
        total_reachable(&a, r),
        count_op(&a, r, OpKind::Sqrt),
        count_op(&a, r, OpKind::Dwrt),
    );
    let (dl, dr) =
        lower_dwrt_owned(arena, root).expect("dwrt lowering must succeed on glyph kernels");
    eprintln!(
        "  lower_dwrt-only baseline: total={} sqrt={}",
        total_reachable(&dl, dr),
        count_op(&dl, dr, OpKind::Sqrt),
    );
    lower_dwrt_owned(&a, r).expect("dwrt lowering must succeed on glyph kernels")
}

/// A closed polygon of `n` straight edges: no curves, so every piece's
/// sliver columns are zero and its implicit is the identity.
fn polygon(points: &[[f32; 2]]) -> Outline {
    let segments = (0..points.len())
        .map(|i| Segment::Line {
            from: points[i],
            to: points[(i + 1) % points.len()],
        })
        .collect();
    Outline {
        contours: vec![Contour::new(segments).expect("a polygon's own vertices close the loop")],
    }
}

/// A regular `n`-gon on a circle of radius 12 centred at (16, 16): every
/// edge is straight, distinct, and non-degenerate, whatever `n` is.
fn regular_polygon(n: usize) -> Outline {
    let points: Vec<[f32; 2]> = (0..n)
        .map(|i| {
            let t = std::f32::consts::TAU * i as f32 / n as f32;
            [16.0 + 12.0 * t.cos(), 16.0 + 12.0 * t.sin()]
        })
        .collect();
    polygon(&points)
}

/// One `sqrt` per piece for the capsule distance `√(d² + t²)`, and one for
/// each of the three gradient normalisations a piece's distance needs — the
/// chord's across and along projections, and the implicit's own `‖∇f‖`.
const SQRT_PER_PIECE: usize = 4;

/// **A glyph is one body, and the fold says how many times it runs.**
///
/// This test used to assert the opposite property: that every affine edge
/// function's gradient `√(DX² + DY²)` folded to a compile-time constant,
/// leaving one `sqrt` per edge, the capsule distance. That property is
/// *given up* deliberately (S1b of
/// docs/plans/2026-09-09-glyph-as-a-fold-execution.md). An edge function's
/// coefficients are table reads now, so `‖∇d‖` is a value rather than a
/// literal and no folding can reach it — and folding it back by
/// precomputing the magnitude on the host would put the distance in the
/// outline's units instead of the lattice's, which is wrong under a
/// magnifying `Kernel::at`.
///
/// What replaces it is the property the fold buys, which the per-edge form
/// could not have stated at all:
///
/// - the arena a glyph **builds** is the same size whatever the outline is
///   — one body per fold, not one fragment per edge, so construction stops
///   being a function of the piece count;
/// - the optimizer's output stays **linear** in the piece count with a
///   fixed budget of [`SQRT_PER_PIECE`] per piece, so a rewrite that
///   multiplied work per pixel still shows up as a hard number;
/// - and `Dwrt` is still fully resolved, which now also covers
///   `passes::lower_dwrt`'s rule that a table read whose index does not move
///   with the differentiation variable is a constant. Without that rule this
///   kernel does not lower at all.
#[test]
fn a_glyph_is_one_body_and_a_fixed_budget_per_piece() {
    let (small, large) = (5usize, 11usize);
    let build = |n: usize| linked(&loop_blinn::glyph(&regular_polygon(n)).kernel());
    let (few, few_root) = build(small);
    let (many, many_root) = build(large);

    // The body is one. A different piece count changes the fold's extent
    // (a `Const`) and the table's height, never the arena's shape.
    assert_eq!(
        (
            total_reachable(&few, few_root),
            count_op(&few, few_root, OpKind::Sqrt),
            count_op(&few, few_root, OpKind::Dwrt),
        ),
        (
            total_reachable(&many, many_root),
            count_op(&many, many_root, OpKind::Sqrt),
            count_op(&many, many_root, OpKind::Dwrt),
        ),
        "a {small}-gon and a {large}-gon must build the same arena: the piece \
         count is data in a table, not structure in the graph"
    );

    for (n, arena, root) in [(small, &few, few_root), (large, &many, many_root)] {
        let (opt, opt_root) = bake_pipeline(arena, root, [32, 32]);
        let opt_sqrt = count_op(&opt, opt_root, OpKind::Sqrt);
        let opt_dwrt = count_op(&opt, opt_root, OpKind::Dwrt);
        eprintln!(
            "{n}-gon: raw total={} sqrt={} dwrt={} -> optimized total={} sqrt={opt_sqrt} \
             dwrt={opt_dwrt}",
            total_reachable(arena, root),
            count_op(arena, root, OpKind::Sqrt),
            count_op(arena, root, OpKind::Dwrt),
            total_reachable(&opt, opt_root),
        );
        assert_eq!(opt_dwrt, 0, "Dwrt must be fully resolved by bake time");
        assert!(
            opt_sqrt <= SQRT_PER_PIECE * n,
            "a {n}-gon's unrolled kernel may keep {SQRT_PER_PIECE} sqrt per piece \
             (the capsule distance and three gradient normalisations); {opt_sqrt} \
             survived, so something is computing a root per pixel that the one \
             body does not ask for"
        );
    }
}

/// Every op a glyph kernel contains must be representable in the e-graph —
/// a gap here silently turns the whole runtime optimization tier into a
/// no-op for glyphs (exactly what happened when `BitAnd`, the mask
/// combinator, was missing from `op_from_kind`).
#[test]
fn lowered_glyph_ops_are_all_egraph_representable() {
    let font = Font::parse(FONT_DATA).unwrap();
    let glyph = font.glyph_kernel_scaled('g', 16.0).expect("glyph");
    let (arena, root) = linked(&glyph.kernel());
    let (lowered, lroot) = lower_dwrt_owned(&arena, root).expect("lower");
    let mut missing = std::collections::BTreeSet::new();
    let len = lowered.nodes_raw().len();
    let mut seen = vec![false; len];
    let mut stack = vec![lroot];
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        let kind = match lowered.node(id) {
            ExprNode::Unary(k, _) => Some(*k),
            ExprNode::Binary(k, _, _) => Some(*k),
            ExprNode::Ternary(k, _, _, _) => Some(*k),
            ExprNode::Param(i) => {
                missing.insert(format!("Param({i})"));
                None
            }
            _ => None,
        };
        if let Some(k) = kind {
            if !pixelflow_search::runtime::is_egraph_representable(k) {
                missing.insert(format!("{k:?}"));
            }
        }
        stack.extend(lowered.children(id));
    }
    assert!(
        missing.is_empty(),
        "ops unconvertible to the e-graph: {missing:?} — the runtime \
         optimizer bails out entirely on any kernel containing them"
    );
}
