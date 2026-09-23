//! Optimization-quality guards for the glyph coverage kernels.
//!
//! A glyph kernel is the hottest runtime-composed arena in the system (every
//! bake evaluates every reachable node per pixel). A glyph is **one fold
//! over one coefficient table** — a `sum_over` of each piece's area term,
//! one fixed body reading its numbers by column at the fold's binder — and
//! the body is written as an integral the e-graph closes
//! (`fonts/loop_blinn.rs`). So the two things worth pinning are that the
//! built arena does not grow with the outline, and that the closed body the
//! optimizer hands the emitter has a fixed budget per piece.
//!
//! These tests count surviving operations through the runtime pipeline
//! (`optimize_runtime_arena`, which saturates, extracts and resolves) — the
//! exact stages `Lattice::bake` runs — so a regression in closure, CSE or
//! extraction shows up as a hard number, not a benchmark whisper.

use pixelflow_graphics::fonts::{loop_blinn, Contour, Font, Outline, Segment};
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use pixelflow_ir::passes::lower_dwrt_owned;
use pixelflow_ir::OpKind;

const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// Count reachable nodes matching `pred` from `root`.
fn count_reachable(arena: &ExprArena, root: ExprId, pred: impl Fn(&ExprNode) -> bool) -> usize {
    let len = arena.len();
    let mut seen = vec![false; len];
    let mut stack = vec![root];
    let mut count = 0;
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        if pred(&arena.node(id)) {
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

/// The same optimization stages `Lattice::bake` runs, and the (arena, root)
/// the emitter would actually schedule.
///
/// # Panics
///
/// When the runtime tier declines the glyph. That used to fall back to the
/// arena as written — which is exactly what a bake would compile then, a
/// glyph whose integrals are left to one-point quadrature, so a fallback
/// here measured the failure and passed.
fn bake_pipeline(arena: &ExprArena, root: ExprId, shape: [u32; 2]) -> (ExprArena, ExprId) {
    let optimized = pixelflow_search::runtime::optimize_runtime_arena(
        arena,
        root,
        pixelflow_ir::LatticeShape::new(shape),
    )
    .unwrap_or_else(|| {
        panic!("the runtime tier declined the glyph; a bake would compile it unoptimized")
    });
    let (a, r) = &*optimized;
    eprintln!(
        "  post-egraph: total={} sqrt={} div={} recip={} dwrt={}",
        total_reachable(a, *r),
        count_op(a, *r, OpKind::Sqrt),
        count_op(a, *r, OpKind::Div),
        count_op(a, *r, OpKind::Recip),
        count_op(a, *r, OpKind::Dwrt),
    );
    (a.clone(), *r)
}

/// A closed polygon of `n` straight edges: no curves, so every piece's bend
/// is zero.
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

/// Square roots a row may keep: a budget per piece, not a count of the
/// body. See the test below.
const SQRT_PER_PIECE: usize = 4;

/// **A glyph is one body, and the fold says how many times it runs.**
///
/// - The arena a glyph **builds** is the same size whatever the outline is
///   — one body, not one fragment per edge, so construction stops being a
///   function of the piece count.
/// - The optimizer's output stays **linear** in the fold's trip count with
///   a fixed budget of [`SQRT_PER_PIECE`] per row, so a rewrite that
///   multiplied work per pixel still shows up as a hard number. The trip
///   count is the piece count rounded up to a bucket
///   (`docs/plans/2026-09-09-glyph-as-a-fold-execution.md` §S3), so a
///   `5`-gon budgets against `8` rows and an `11`-gon against `16` — the
///   padding rows are exact identities of the fold (their gate is
///   `loop_blinn::tests::a_padding_row_is_an_exact_identity_of_the_fold`)
///   but they are still rows the fold runs and this budget still counts;
/// - and no `Dwrt` reaches the emitter.
#[test]
fn a_glyph_is_one_body_and_a_fixed_budget_per_piece() {
    let (small, large) = (5usize, 11usize);
    let build = |n: usize| {
        loop_blinn::glyph(&regular_polygon(n))
            .kernel()
            .linked_parts()
    };
    let (few, few_root) = build(small);
    let (many, many_root) = build(large);

    // The body is one. A different piece count changes the fold's extent
    // and the table's height, never the arena's shape.
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
            "{n}-gon: raw total={} sqrt={} -> optimized total={} sqrt={opt_sqrt} \
             dwrt={opt_dwrt}",
            total_reachable(arena, root),
            count_op(arena, root, OpKind::Sqrt),
            total_reachable(&opt, opt_root),
        );
        assert_eq!(opt_dwrt, 0, "Dwrt must be fully resolved by bake time");
        // The fold's trip count is `n` rounded up to a bucket
        // (`loop_blinn::bucketed_trip_count`, mirrored here rather than
        // exposed: it is `u32::next_power_of_two`, not a bespoke rule), not
        // `n` itself — see the budget's own doc above.
        let bucketed_rows = (n as u32).next_power_of_two() as usize;
        assert!(
            opt_sqrt <= SQRT_PER_PIECE * bucketed_rows,
            "a {n}-gon's kernel (bucketed to {bucketed_rows} rows) may keep \
             {SQRT_PER_PIECE} sqrt per row; {opt_sqrt} survived, so something \
             is computing a root per pixel that the one body does not ask for"
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
    let (arena, root) = glyph.kernel().linked_parts();
    let (lowered, lroot) = lower_dwrt_owned(&arena, root).expect("lower");
    let mut missing = std::collections::BTreeSet::new();
    let len = lowered.len();
    let mut seen = vec![false; len];
    let mut stack = vec![lroot];
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        let kind = match lowered.node(id) {
            ExprNode::Unary(k, _) => Some(k),
            ExprNode::Binary(k, _, _) => Some(k),
            ExprNode::Ternary(k, _, _, _) => Some(k),
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
