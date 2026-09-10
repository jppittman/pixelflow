//! Optimization-quality guards for the glyph winding kernels.
//!
//! The winding kernels are the hottest runtime-composed arenas in the system
//! (every glyph bake evaluates every reachable node per pixel), and their
//! cost is dominated by derivative-bearing subtrees:
//!
//! - `AnalyticalLine`: `d = X − ((Y−y0)·k + x0)` with baked constants, so
//!   `DX(d) = 1` and `DY(d) = −k` — the whole gradient `√(DX²+DY²)` is a
//!   compile-time constant. If the optimizer doesn't fold it, every pixel
//!   pays a sqrt for a number known at bake time.
//! - `AnalyticalQuad`: the value path and both `Dwrt` gradient paths share
//!   the discriminant / `sqrt(disc)` subtrees; losing that sharing multiplies
//!   the per-pixel sqrt count.
//!
//! These tests count surviving operations through the runtime pipeline
//! (`optimize_runtime_term` → `lower_dwrt`) — the exact stages
//! `Lattice::bake` runs — so a regression in derivative folding or CSE shows
//! up as a hard number, not a benchmark whisper.

use pixelflow_graphics::fonts::ttf_curve_analytical::{AnalyticalLine, AnalyticalQuad};
use pixelflow_ir::expr::{Environment, Term};
use pixelflow_ir::passes::lower_dwrt;
use pixelflow_ir::{ExprData, Node, OpKind, Rooted};

/// Count reachable nodes matching `pred` from `root`.
fn count_reachable(root: Node<'_, ExprData>, pred: impl Fn(Node<'_, ExprData>) -> bool) -> usize {
    root.descendants().filter(|n| pred(*n)).count()
}

fn count_op(root: Node<'_, ExprData>, op: OpKind) -> usize {
    count_reachable(root, |n| matches!(*n, ExprData::Op(k) if k == op))
}

fn total_reachable(root: Node<'_, ExprData>) -> usize {
    root.descendants().count()
}

/// Run the same optimization stages `Lattice::bake` runs, then lower any
/// residual `Dwrt` exactly as the compile entries do, and report the final
/// term the emitter would actually schedule. Prints per-stage counts so a
/// failure localizes to the stage that dropped the ball.
fn bake_pipeline(term: Term<'_>) -> (Rooted<ExprData>, Environment) {
    let optimized =
        pixelflow_search::runtime::optimize_runtime_term(term, pixelflow_ir::LatticeShape::POINT);
    let dl = lower_dwrt(term).expect("dwrt lowering must succeed on winding kernels");
    let dl_term = Term::new(dl.entry(), term.env());
    eprintln!(
        "  lower_dwrt-only baseline: total={} sqrt={}",
        total_reachable(dl_term.root()),
        count_op(dl_term.root(), OpKind::Sqrt),
    );
    match optimized.as_deref() {
        Some((a, env)) => {
            let opt_term = Term::new(a.entry(), env);
            eprintln!(
                "  post-egraph: total={} sqrt={} dwrt={}",
                total_reachable(opt_term.root()),
                count_op(opt_term.root(), OpKind::Sqrt),
                count_op(opt_term.root(), OpKind::Dwrt),
            );
            let lowered =
                lower_dwrt(opt_term).expect("dwrt lowering must succeed on winding kernels");
            (lowered, env.clone())
        }
        None => {
            eprintln!(
                "  post-egraph: total={} sqrt={} dwrt={} (declined; same as raw)",
                total_reachable(term.root()),
                count_op(term.root(), OpKind::Sqrt),
                count_op(term.root(), OpKind::Dwrt),
            );
            (dl, term.env().clone())
        }
    }
}

#[test]
fn line_gradient_folds_to_a_constant() {
    let line = AnalyticalLine::from_points([2.0, 1.0], [10.0, 30.0]).expect("non-degenerate");
    let kernel = line.kernel();
    let term = kernel.term();

    let raw_sqrt = count_op(term.root(), OpKind::Sqrt);
    let raw_dwrt = count_op(term.root(), OpKind::Dwrt);
    let raw_total = total_reachable(term.root());

    let (opt, opt_env) = bake_pipeline(term);
    let opt_term = Term::new(opt.entry(), &opt_env);
    let opt_sqrt = count_op(opt_term.root(), OpKind::Sqrt);
    let opt_dwrt = count_op(opt_term.root(), OpKind::Dwrt);
    let opt_total = total_reachable(opt_term.root());

    eprintln!(
        "line: raw total={raw_total} sqrt={raw_sqrt} dwrt={raw_dwrt} -> \
         optimized total={opt_total} sqrt={opt_sqrt} dwrt={opt_dwrt}"
    );

    assert_eq!(opt_dwrt, 0, "Dwrt must be fully resolved by bake time");
    // DX(d) = 1 and DY(d) = -dx_over_dy (both constants), so
    // sqrt(DX² + DY²) is a compile-time constant: no sqrt may survive.
    assert_eq!(
        opt_sqrt, 0,
        "the line kernel's gradient is a compile-time constant; a surviving \
         sqrt means every pixel of every straight glyph edge pays for a \
         number known at bake time"
    );
}

#[test]
fn quad_shares_the_discriminant_between_value_and_gradient() {
    // A genuinely quadratic segment (control point off the chord).
    let quad = AnalyticalQuad::new([0.0, 0.0], [8.0, 20.0], [16.0, 0.0]);
    let kernel = quad.kernel();
    let term = kernel.term();

    let raw_sqrt = count_op(term.root(), OpKind::Sqrt);
    let raw_dwrt = count_op(term.root(), OpKind::Dwrt);
    let raw_total = total_reachable(term.root());

    let (opt, opt_env) = bake_pipeline(term);
    let opt_term = Term::new(opt.entry(), &opt_env);
    let opt_sqrt = count_op(opt_term.root(), OpKind::Sqrt);
    let opt_dwrt = count_op(opt_term.root(), OpKind::Dwrt);
    let opt_total = total_reachable(opt_term.root());

    eprintln!(
        "quad: raw total={raw_total} sqrt={raw_sqrt} dwrt={raw_dwrt} -> \
         optimized total={opt_total} sqrt={opt_sqrt} dwrt={opt_dwrt}"
    );

    assert_eq!(opt_dwrt, 0, "Dwrt must be fully resolved by bake time");
    // The raw kernel holds ONE sqrt(disc) (shared by both roots) plus the
    // two gradient sqrts (one per root's ramp). Differentiating d(Y) chains
    // through sqrt(disc), whose derivative reuses the SAME sqrt node
    // (d/dY √u = u′/(2√u)) — so a fully-shared result still has exactly
    // three sqrts: sqrt(disc), and the two gradient-magnitude sqrts.
    // Every sqrt beyond that is lost sharing, paid per pixel per curve.
    assert!(
        opt_sqrt <= 3,
        "expected ≤3 surviving sqrts (shared disc + two gradient ramps), \
         got {opt_sqrt}: the value/gradient paths have stopped sharing the \
         discriminant"
    );
}

/// Every op the lowered winding kernels contain must be representable in
/// the e-graph — a gap here silently turns the whole runtime optimization
/// tier into a no-op for glyphs (exactly what happened when `BitAnd`, the
/// Y-range mask combinator, was missing from `op_from_kind`).
#[test]
fn lowered_winding_ops_are_all_egraph_representable() {
    let line = AnalyticalLine::from_points([2.0, 1.0], [10.0, 30.0]).expect("non-degenerate");
    let kernel = line.kernel();
    let term = kernel.term();
    let lowered = lower_dwrt(term).expect("lower");
    let mut missing = std::collections::BTreeSet::new();
    for node in lowered.entry().descendants() {
        match *node {
            ExprData::Op(k) => {
                if !pixelflow_search::runtime::is_egraph_representable(k) {
                    missing.insert(format!("{k:?}"));
                }
            }
            ExprData::Param(i) => {
                missing.insert(format!("Param({i})"));
            }
            _ => {}
        }
    }
    assert!(
        missing.is_empty(),
        "ops unconvertible to the e-graph: {missing:?} — the runtime \
         optimizer bails out entirely on any kernel containing them"
    );
}

/// Optimization must preserve a real glyph's coverage within float-
/// reassociation noise. The JIT-vs-interpreter goldens deliberately compare
/// the compiler against the interpreter on the SAME (optimized) arena, so
/// this is the guard that pins optimized-vs-raw — without it, an unsound
/// rewrite would slip past the goldens by corrupting both sides equally.
///
/// Tolerance: reassociation/FMA-fusion re-rounds a long winding sum at the
/// 1e-4 scale (observed); an unsound rule (wrong branch of a root, a lost
/// mask) shifts coverage by O(1).
///
/// **Sizes, not one size, and coarse ones.** Cost is quadratic in the size
/// while the chance of catching this is roughly linear in the row count, so a
/// wide sweep of coarse sizes dominates a narrow sweep that includes fine
/// ones — 128 px alone would be 75% of the texels and has never caught
/// anything.
///
/// This is **more expensive than what it replaced**, and worth it. Against the
/// single 32 px pass this test used to make, ten sizes cost about 7x more
/// (7200 texels per glyph against 1024) — roughly 40 s in a debug run. It buys
/// 29 real divergences at four sizes where 32 px alone found none. An earlier
/// revision of this comment claimed a 6.9x *saving*; that compared against an
/// intermediate 7/12/17/32/64/128 sweep that existed only within this branch,
/// never on `main`, which is not a baseline anyone else would recognize.
const SIZES: [u32; 10] = [7, 9, 11, 13, 15, 17, 19, 21, 23, 32];

/// **Sizes, not one size.** A rounding difference only *decides* something
/// where a comparison sits on a knife edge, and which rows land on one is a
/// function of the size, so a single size is not a sample of this failure
/// mode — it is a lottery ticket. `'8'` at 17 px is the recorded case: the
/// quadratic solver's `disc >= 0` is exact zero at the parabola's vertex row,
/// one rounding of `Y·slope + c` (fused) against two (raw) flips it, and a
/// whole crossing (0.5 of coverage) appears on one side and not the other.
/// Eight texels of a single row moved. 32 px — the only size this test used to
/// run at, and the size the glyph goldens use — sees nothing.
#[test]
fn optimized_glyph_matches_raw_within_reassociation_noise() {
    use pixelflow_graphics::fonts::Font;
    use pixelflow_ir::binding::BindingTable;
    use pixelflow_ir::eval_scalar;

    /// Reassociation and FMA fusion re-round a long winding sum at the 1e-4
    /// scale; an unsound rewrite moves coverage by O(1).
    const TOLERANCE: f32 = 1e-3;

    /// There used to be 29 of these, all on the `'8'` waist at 13/15/17/21
    /// px: the quadratic solver's `disc >= 0` is a knife edge at a tangency,
    /// and the two arenas' different fusion choices landed on different sides
    /// of it.
    ///
    /// They are gone, and not because the tangency got any less sharp — it is
    /// still a knife edge, and `quad_tangency_winding` and `freetype_oracle`
    /// still hold that defect under test. What changed is that there is only
    /// one optimizer now. The macro tier used to saturate the glyph's
    /// fragments on the *AST* before the runtime tier ever saw them, so two
    /// independent sets of fusion decisions compounded, and it was the
    /// compounding that straddled the edge. `kernel!` now emits the arena it
    /// lowered and the runtime tier is the only thing that rewrites it.
    ///
    /// So this is back to the guard it was always meant to be: optimization
    /// must not move coverage anywhere, and the assertion is emptiness.
    const KNOWN_DIVERGENT_TEXELS: usize = 0;

    const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");
    let font = Font::parse(FONT_DATA).unwrap();
    let mut divergences: Vec<String> = Vec::new();

    for size in SIZES {
        for ch in ['A', 'O', 'g', '8'] {
            let kernel = font
                .glyph_kernel_scaled(ch, size as f32)
                .expect("glyph kernel");
            let term = kernel.term();
            let raw = lower_dwrt(term).expect("lower raw");
            let raw_term = Term::new(raw.entry(), term.env());
            // The lattice a bake of this glyph would compile at — the
            // extraction is a function of the shape, and the bake's is the one
            // that reaches pixels.
            let extent = size + size / 2;
            let optimized = pixelflow_search::runtime::optimize_runtime_term(
                term,
                pixelflow_ir::LatticeShape::new([extent, extent]),
            )
            .expect("glyph arenas must optimize (pure arithmetic + Dwrt + masks)");
            let opt_term = Term::new(optimized.0.entry(), &optimized.1);

            for j in 0..extent as usize {
                for i in 0..extent as usize {
                    let (x, y) = (i as f32 + 0.5, j as f32 + 0.5);
                    let want = eval_scalar(raw_term, &[x, y], &BindingTable::empty());
                    let got = eval_scalar(opt_term, &[x, y], &BindingTable::empty());
                    // Before the comparison, not folded into it: `NaN >= x` is
                    // false, so a threshold test *accepts* a non-finite
                    // coverage silently. The `assert!` this loop replaced
                    // caught NaN for free by asserting the negation; a
                    // collector has to say so itself.
                    assert!(
                        want.is_finite() && got.is_finite(),
                        "{ch}@{size} texel ({i},{j}): non-finite coverage \
                         (raw {want}, optimized {got})"
                    );
                    if (want - got).abs() >= TOLERANCE {
                        divergences.push(format!(
                            "{ch}@{size} texel ({i},{j}): raw {want} vs optimized {got} \
                             (delta {})",
                            got - want
                        ));
                    }
                }
            }
        }
    }

    // Reported together rather than at the first hit: which rows sit on a
    // knife edge is the diagnostic, and one texel does not show it.
    //
    // Both saturation and `eval_scalar` are deterministic (CLAUDE.md: a kernel
    // cannot be built differently on two machines), so this is stable across
    // targets. If a platform reports a divergence, that is a finding about
    // determinism or about a rewrite, never a flaky test.
    assert!(
        divergences.is_empty(),
        "optimization changed glyph coverage:\n{}",
        divergences.join("\n")
    );
    assert_eq!(
        divergences.len(),
        KNOWN_DIVERGENT_TEXELS,
        "optimization must not move glyph coverage; got {} divergent \
         texels.\n{}",
        divergences.len(),
        divergences.join("\n")
    );
}
