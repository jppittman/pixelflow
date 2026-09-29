//! An integral written in `kernel!` is the integral the builder builds, and
//! it means what it says (docs/plans/2026-09-25-the-language-is-kernel.md
//! §1.5, B4).
//!
//! **What it is.** `area(|u, v| e)` is the prelude's pixel, exactly
//! `integral(-H..H, |v| integral(-H..H, |u| e))` with `H` the IR's
//! `PIXEL_HALF_WIDTH`, and the author writes the shift:
//! `area(|u, v| f(X + u, Y + v))` is the builder's `f.area()`. The first
//! pins below hold the two to one `pixelflow_ir::key::canonical` for a
//! half-plane and a disc, and `monotone_root` to `integral::monotone_root`
//! under `ROOT_FLOOR`. `kernel_raw!` keeps the lowered shape, so each comparison
//! is the front end's word and not the optimizer's. An integral's variable
//! is bound inside-out, as a fold's index is, so a nested integral captures
//! nothing.
//!
//! **What it means.** Whether an integral closes is the e-graph's
//! (`FactorFold`, `NarrowInterval`, `ClampMoment`, `ArcMoment`); one it
//! leaves open is legalized by quadrature. The value pins read closed forms,
//! and each first asserts the extraction left no integral for quadrature
//! (`unclosed_integrals`), since the one-point quadrature would pass a pin
//! that samples a pixel's centre. The judge is `f64` arithmetic written
//! here, never a pixelflow evaluator.

use pixelflow_compiler::{kernel, kernel_raw};
use pixelflow_core::{Kernel, Lattice, Uniform};
use pixelflow_ir::arena::UniformDecl;
use pixelflow_ir::integral::{self, ROOT_FLOOR, Rise, RootFloor};
use pixelflow_ir::key::canonical;
use pixelflow_ir::{ExprArena, ExprId, ExprNode, Fold, LatticeShape};
use pixelflow_search::runtime::{optimize_runtime_arena, unclosed_integrals};

/// Whether two kernels are one program: the same canonical key, the shape
/// the JIT compiles under, with its uniforms numbered by first occurrence —
/// and the same value in each uniform's slot. A uniform's identity is minted
/// per call, so two calls' kernels differ in it and nothing else.
fn assert_same_program(written: &Kernel, built: &Kernel) {
    let (written_arena, written_root) = written.parts();
    let (built_arena, built_root) = built.parts();
    let written_form = canonical(written_arena, written_root);
    let built_form = canonical(built_arena, built_root);
    assert_eq!(
        written_form.key,
        built_form.key,
        "written: {}\nbuilt:   {}",
        written_arena.display(written_root),
        built_arena.display(built_root),
    );
    let values = |uniforms: &[UniformDecl]| -> Vec<f32> {
        uniforms.iter().map(|uniform| uniform.default).collect()
    };
    assert_eq!(
        values(&written_form.uniforms),
        values(&built_form.uniforms),
        "each uniform slot holds the same value"
    );
}

fn x() -> Kernel {
    Kernel::x()
}

fn y() -> Kernel {
    Kernel::y()
}

fn constant(v: f32) -> Kernel {
    Kernel::constant(v)
}

/// `[mask]`, as the builder's glyph spells it: a choice, never a product.
fn indicator(mask: &Kernel) -> Kernel {
    mask.select(&constant(1.0), &constant(0.0))
}

// ─────────────────────────── one program ───────────────────────────

/// Where the half-plane's edge is: off the lattice's integers and their
/// halves, so every texel's area is a fraction of the pixel.
const EDGE: f32 = 3.25;
/// The disc's radius.
const RADIUS: f32 = 2.5;

kernel_raw! {
    /// `EDGE`, written again: a body captures nothing from the host.
    const C: f32 = 3.25;

    fn indicator(m: bool) -> f32 { if m { 1.0 } else { 0.0 } }

    /// The half-plane left of `C`.
    fn left_of(x: f32) -> f32 { indicator(x < C) }

    /// The disc of radius `r` about `(cx, cy)`.
    fn in_disc(x: f32, y: f32, cx: f32, cy: f32, r: f32) -> f32 {
        let dx = x - cx;
        let dy = y - cy;
        indicator(dx * dx + dy * dy < r * r)
    }

    pub fn half_plane_area() -> f32 { area(|u, v| left_of(X + u)) }

    pub fn disc_area(cx: f32, cy: f32, r: f32) -> f32 {
        area(|u, v| in_disc(X + u, Y + v, cx, cy, r))
    }

    /// The same disc with the variables transposed: `u` shifting `Y`.
    pub fn disc_area_transposed(cx: f32, cy: f32, r: f32) -> f32 {
        area(|u, v| in_disc(X + v, Y + u, cx, cy, r))
    }
}

/// The half-plane `[X < c]` over the pixel is `Kernel::area` of it.
#[test]
fn the_area_of_a_half_plane_is_kernel_area() {
    let built = indicator(&x().lt(&constant(EDGE))).area();
    assert_same_program(&half_plane_area(), &built);
}

/// The disc `[(X − cx)² + (Y − cy)² < r²]` over the pixel is `Kernel::area`
/// of it: `v` shifts `Y`, `u` shifts `X`, and the `v` integral is outermost.
/// An entry's parameters are its uniforms (plan §1.4), so the builder's
/// disc reads `cx`, `cy` and `r` as uniforms too.
#[test]
fn the_area_of_a_disc_is_kernel_area() {
    let (cx, cy) = (1.5, -0.75);
    let uniform = |value: f32| Uniform::new(value).kernel();
    let (dx, dy) = (x().sub(&uniform(cx)), y().sub(&uniform(cy)));
    let r = uniform(RADIUS);
    let built = indicator(&dx.mul(&dx).add(&dy.mul(&dy)).lt(&r.mul(&r))).area();
    assert_same_program(&disc_area(cx, cy, RADIUS), &built);

    // Transposed, it is another program: which variable shifts which
    // coordinate is part of what `area` is.
    let transposed = disc_area_transposed(cx, cy, RADIUS);
    let (written, written_root) = transposed.parts();
    let (built, built_root) = built.parts();
    assert_ne!(
        canonical(written, written_root).key,
        canonical(built, built_root).key
    );
}

// An arc written as the area it bounds, with literal coefficients: rising
// in both coordinates, run downward (`S = −1`), counted positively
// (`σ = +1`) and cut to the rows `[−4, 4]` it reaches — the integrand the
// font's pieces were written with before each became its closed form
// (`pixelflow-graphics/src/fonts/loop_blinn.rs`), kept as the arc integral
// the rules close (`a_glyph_piece_closes`, below).
kernel_raw! {
    const X0: f32 = 1.25;
    const E0X: f32 = 1.5;
    const E1X: f32 = 0.5;
    const Y0: f32 = -3.0;
    const E0Y: f32 = 2.0;
    const E1Y: f32 = 3.5;
    const SIGMA: f32 = 1.0;
    const S: f32 = -1.0;
    const ROWS_LO: f32 = -4.0;
    const ROWS_HI: f32 = 4.0;

    fn indicator(m: bool) -> f32 { if m { 1.0 } else { 0.0 } }

    /// χ: the region left of the arc, within its band.
    fn left_of_the_arc(x: f32, y: f32) -> f32 {
        let b = E0Y.max(0.0);
        let bx = E0X.max(0.0);
        let a = E1Y.max(0.0) - b;
        let ax = E1X.max(0.0) - bx;
        let t = monotone_root(y - Y0, b, a);
        let x_at_t = X0 + t * (bx + bx + ax * t);
        indicator(0.0 <= t) * indicator(t < 1.0) * indicator(x < x_at_t)
    }

    /// σ·∫∫χ over the pixel about `(X, S·Y)`, cut to the rows the piece
    /// reaches.
    pub fn piece_term() -> f32 {
        let term = SIGMA * area(|u, v| left_of_the_arc(X + u, S * Y + v));
        if (Y > ROWS_LO) & (Y < ROWS_HI) { term } else { 0.0 }
    }
}

/// `monotone_root` as the builder spells it: the one definition, over an
/// arena the operands are spliced into, under `ROOT_FLOOR`.
fn monotone_root(delta: &Kernel, step: &Kernel, bend: &Kernel) -> Kernel {
    fn graft(arena: &mut ExprArena, k: &Kernel) -> ExprId {
        let (from, root) = k.parts();
        arena.splice(from, root)
    }
    let mut arena = ExprArena::new();
    let delta = graft(&mut arena, delta);
    let rise = Rise {
        step: graft(&mut arena, step),
        bend: graft(&mut arena, bend),
    };
    let floor = RootFloor::new(ROOT_FLOOR).expect("the largest floor RootFloor admits");
    let root = integral::monotone_root(&mut arena, delta, rise, floor);
    Kernel::from_parts(arena, root)
}

/// `monotone_root(δ, step, bend)` is `integral::monotone_root` under
/// `ROOT_FLOOR`, on the same operands.
#[test]
fn monotone_root_is_the_one_definition() {
    let written = kernel_raw!(|| monotone_root(Y - 2.0, X.max(0.0), X * 0.5));
    let built = monotone_root(
        &y().sub(&constant(2.0)),
        &x().max(&constant(0.0)),
        &x().mul(&constant(0.5)),
    );
    assert_same_program(&written, &built);
}

// ─────────────────────────── binders ───────────────────────────

/// The `Var` index of the binder of every `Reduce` reachable from `root`,
/// each fold before the folds in its body.
fn binders(arena: &ExprArena, root: ExprId) -> Vec<u8> {
    let mut seen = vec![false; arena.len()];
    let mut stack = vec![root];
    let mut out = Vec::new();
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        if let ExprNode::Reduce { fold, .. } = arena.node(id) {
            out.push(fold.binder().var());
        }
        stack.extend(arena.children(id));
    }
    out
}

/// `∫_u ∫_v f(u, v)`: the inner integral takes slot 0 and the outer slot 1,
/// and the body reads both. It is not `∫_u ∫_v f(v, v)` — the capture the
/// builder once had (`kernel.rs`'s `BinderScope`) — nor `∫_u ∫_v f(u, u)`.
#[test]
fn a_nested_integral_captures_nothing() {
    let written = kernel_raw!(|| integral(0..1, |u| integral(0..1, |v| u * 2.0 + v)));
    let captured_inner = kernel_raw!(|| integral(0..1, |u| integral(0..1, |v| v * 2.0 + v)));
    let captured_outer = kernel_raw!(|| integral(0..1, |u| integral(0..1, |v| u * 2.0 + u)));
    let (arena, root) = written.parts();
    for captured in [&captured_inner, &captured_outer] {
        let (other, other_root) = captured.parts();
        assert_ne!(canonical(arena, root), canonical(other, other_root));
    }
    assert_eq!(binders(arena, root), [5, 4], "outer slot 1, inner slot 0");
    let ExprNode::Reduce {
        fold: Fold::Interval(_),
        body,
    } = arena.node(root)
    else {
        panic!("an integral: {}", arena.display(root));
    };
    let ExprNode::Reduce {
        fold: Fold::Interval(_),
        body,
    } = arena.node(body)
    else {
        panic!("an integral in it: {}", arena.display(body));
    };
    let ExprNode::Binary(_, twice_u, v) = arena.node(body) else {
        panic!("the integrand: {}", arena.display(body));
    };
    assert!(
        matches!(arena.node(v), ExprNode::Var(4)),
        "v is the inner binder"
    );
    let ExprNode::Binary(_, u, _) = arena.node(twice_u) else {
        panic!("u · 2: {}", arena.display(twice_u));
    };
    assert!(
        matches!(arena.node(u), ExprNode::Var(5)),
        "u is the outer binder"
    );
}

/// An integral inside a fold, and a fold inside an integral, bind distinct
/// slots, inside-out, whichever encloses which.
#[test]
fn integrals_and_folds_bind_distinct_slots() {
    let fold_of_integrals =
        kernel_raw!(|| (0..3).map(|i| integral(0.0..1.0, |u| u * (i as f32))).sum());
    let (arena, root) = fold_of_integrals.parts();
    assert_eq!(binders(arena, root), [5, 4]);
    let integral_of_folds =
        kernel_raw!(|| integral(0.0..1.0, |u| (0..3).map(|i| u * (i as f32)).sum()));
    let (arena, root) = integral_of_folds.parts();
    assert_eq!(binders(arena, root), [5, 4]);
}

kernel_raw! {
    /// Integrals in a fold over a structural range: a template, whose range
    /// its host function fills in per instantiation (§1.4).
    pub fn integrals_over_n<const N: usize>() -> f32 {
        (0..N).map(|i| integral(0.0..1.0, |u| u * (i as f32) + X)).sum()
    }

    /// The same, over a range known at expansion.
    pub fn integrals_over_three() -> f32 {
        (0..3).map(|i| integral(0.0..1.0, |u| u * (i as f32) + X)).sum()
    }
}

/// An integral inside a template's open fold is the integral a known
/// range's fold holds: instantiated at `N = 3`, the template is the program
/// written over `0..3`, the interval and both slots included.
#[test]
fn an_integral_in_a_template_is_the_integral_of_its_instance() {
    let instance = integrals_over_n::<3>();
    assert_same_program(&instance, &integrals_over_three());
    let (arena, root) = instance.parts();
    assert_eq!(binders(arena, root), [5, 4], "the fold at slot 1, u at 0");
}

// ─────────────────────────── what it means ───────────────────────────

/// The lattice the values are read over: 13 is prime, so no SIMD width
/// divides a row and every row ends in a partial batch.
const FRAME: (usize, usize) = (13, 7);

fn frame_shape() -> LatticeShape {
    LatticeShape::new([FRAME.0 as u32, FRAME.1 as u32])
}

/// How many interval folds reachable from `root`.
fn integrals(arena: &ExprArena, root: ExprId) -> usize {
    let mut seen = vec![false; arena.len()];
    let mut stack = vec![root];
    let mut n = 0;
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        n += usize::from(matches!(
            arena.node(id),
            ExprNode::Reduce {
                fold: Fold::Interval(_),
                ..
            }
        ));
        stack.extend(arena.children(id));
    }
    n
}

/// `kernel` closes: the runtime tier's extraction leaves no integral for
/// quadrature, and none reaches what it compiles.
fn assert_closed(name: &str, kernel: &Kernel) {
    let (arena, root) = kernel.parts();
    assert!(integrals(arena, root) > 0, "{name}: nothing to close");
    assert_eq!(
        unclosed_integrals(arena, root, frame_shape()),
        Some(0),
        "{name}: extraction left integrals for quadrature"
    );
    let optimized = optimize_runtime_arena(arena, root, frame_shape())
        .unwrap_or_else(|| panic!("{name}: the runtime tier declined"));
    let (out, out_root) = &*optimized;
    assert_eq!(
        integrals(out, *out_root),
        0,
        "{name}: {}",
        out.display(*out_root)
    );
}

/// The area of the half-plane `x < c` over the pixel about a sample `x` is
/// `clamp(c − x + ½, 0, 1)`, exactly: `NarrowInterval` closes the inner
/// integral to that clamp, and the outer integrates a value its variable
/// does not reach.
#[test]
fn the_area_of_a_half_plane_is_exact() {
    for (name, k) in [
        ("kernel_raw!", half_plane_area()),
        (
            "kernel!",
            kernel!(|| area(|u, v| if X + u < 3.25 { 1.0 } else { 0.0 })),
        ),
    ] {
        assert_closed(name, &k);
        let baked = Lattice::frame(FRAME.0, FRAME.1).bake(&k);
        for (index, &texel) in baked.buffer().iter().enumerate() {
            let column = (index % FRAME.0) as f64;
            let want = (f64::from(EDGE) - column + 0.5).clamp(0.0, 1.0);
            assert_eq!(f64::from(texel), want, "{name}: column {column}");
        }
    }
}

/// `f32`'s unit roundoff, `2⁻²⁴`.
const ROUNDOFF: f64 = 1.0 / 16_777_216.0;
/// How many roundoffs of its terms' magnitude a closed form may lose: the
/// area oracle's bound (`pixelflow-core/tests/area_oracle.rs`).
const ROUNDOFFS: f64 = 8.0;

/// `∫₀¹ clamp(3u/2 − ½, 0, 1) du = ⅓` at every sample, to the area
/// oracle's tolerance: eight roundoffs of the closed form's terms.
///
/// The ramp is zero up to `u = ⅓` and rises to `1` at `u = 1`, so the
/// integral is `∫_{1/3}^{1} (3u/2 − ½) du = ⅓`. `ClampMoment` closes it;
/// the one-point quadrature that legalizes an open integral would read the
/// ramp at the midpoint, `¼`, so `⅓` is the closed form's alone. (`∫₀¹ u²`
/// would say the same, and closes by no rule the e-graph has:
/// `the_power_moment_is_open_and_baked_by_quadrature_today` pins it open.)
#[test]
fn the_integral_of_a_ramp_is_a_third() {
    let (slope, offset, lo, hi) = (1.5_f64, -0.5_f64, 0.0_f64, 1.0_f64);
    let exact = 1.0 / 3.0;
    // The clamp's ends `z₀`, `z₁`, its upper bound, and the length.
    let terms = (slope * lo + offset).abs() + (slope * hi + offset).abs() + 1.0 + (hi - lo);
    let tolerance = ROUNDOFFS * ROUNDOFF * terms;
    for (name, k) in [
        (
            "kernel_raw!",
            kernel_raw!(|| integral(0.0..1.0, |u| (1.5 * u - 0.5).clamp(0.0, 1.0))),
        ),
        (
            "kernel!",
            kernel!(|| integral(0.0..1.0, |u| (1.5 * u - 0.5).clamp(0.0, 1.0))),
        ),
    ] {
        assert_closed(name, &k);
        let baked = Lattice::frame(FRAME.0, FRAME.1).bake(&k);
        for (index, &texel) in baked.buffer().iter().enumerate() {
            let error = (f64::from(texel) - exact).abs();
            assert!(
                error <= tolerance,
                "{name}: texel {index} is {texel}, error {error:e} > {tolerance:e}"
            );
        }
    }
}

/// `∫₀¹ u² du = ⅓` does not close today, and this pins that it does not.
/// No rule the e-graph has closes a power moment — `pixelflow-search`'s
/// `egraph/integral.rs` says the power-moment rule waits for a kernel that
/// needs it — so extraction keeps the integral (`unclosed_integrals` is
/// `Some(1)`, the predicate `assert_closed` reads), and the bake reads the
/// one-point quadrature that legalizes an open integral: the integrand at
/// the interval's midpoint times its length, `(½)² · 1 = ¼`, exactly.
///
/// That is the language as specified, not a wrong answer: whether an
/// integral closes is the e-graph's, and quadrature is what an open one
/// bakes to. It is pinned so it cannot change unseen. **A power-moment rule
/// landing fails this test on purpose**: then change it to assert
/// `assert_closed` and `⅓` to the area oracle's tolerance, as
/// `the_integral_of_a_ramp_is_a_third` does.
#[test]
fn the_power_moment_is_open_and_baked_by_quadrature_today() {
    let midpoint_quadrature = 0.5_f32 * 0.5 * 1.0;
    for (name, k) in [
        ("kernel_raw!", kernel_raw!(|| integral(0.0..1.0, |u| u * u))),
        ("kernel!", kernel!(|| integral(0.0..1.0, |u| u * u))),
    ] {
        let (arena, root) = k.parts();
        assert_eq!(integrals(arena, root), 1, "{name}: one integral written");
        assert_eq!(
            unclosed_integrals(arena, root, frame_shape()),
            Some(1),
            "{name}: a power moment closed; assert `⅓` here now (see this test's doc)"
        );
        let baked = Lattice::frame(FRAME.0, FRAME.1).bake(&k);
        for (index, &texel) in baked.buffer().iter().enumerate() {
            assert_eq!(
                texel, midpoint_quadrature,
                "{name}: texel {index} is not the midpoint quadrature `¼`"
            );
        }
    }
}

/// An arc written as its area in `kernel!` closes: no integral it is
/// written with reaches the emitter.
#[test]
fn a_glyph_piece_closes() {
    assert_closed("piece_term", &piece_term());
}
