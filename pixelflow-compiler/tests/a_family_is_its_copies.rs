//! A family is its copies (docs/plans/2026-09-25-the-language-is-kernel.md
//! §1.6, §1.7; B3).
//!
//! `pieces: [Row; N]` is `N` elements' scalar uniforms at static slots,
//! element-major, and not a table: `pieces.into_iter().map(|p| e).sum()`
//! is `e[p := pieces[0]] + e[p := pieces[1]] + …`, the copies made when
//! `glyph::<N>` is instantiated, each over its own element's uniforms. So
//! the program at `N = 3` is the program with three record parameters
//! summed by hand — one canonical key — and has no fold, no binder and no
//! index for the family. Its count is structural: two glyphs of one `N` are
//! one program, whatever their pieces, and `N = 3` and `N = 4` are two.
//!
//! One key with the terms written out holds at `N ≥ 1`. At `N = 0` a fold
//! around an iteration whose body binds a fold of its own may key apart
//! from the same program written out: lowering chose the enclosing fold's
//! binder around the body, which `N = 0` deletes, so the two differ by a
//! binder's name alone. They are α-equivalent and compute the same pixels;
//! the cost is one degenerate program compiled twice, never a value.
//!
//! The block below is §1.7's, as the plan writes it, with the iteration
//! spelled as it was chosen (§4 Q2): it expands, bakes, keys and rebinds.
//! What an iteration *means* is pinned against rustc in
//! `rustc_is_the_oracle.rs`; the refusals are the parser's and `sema`'s
//! unit tests.

use pixelflow_compiler::{kernel, kernel_raw};
use pixelflow_core::{ArityMismatch, Kernel, Lattice, Manifold};
use pixelflow_ir::key::canonical;
use pixelflow_ir::{ExprArena, ExprId, ExprNode, Fold};

/// §1.7's block, written once and expanded by both macros: `kernel!` below,
/// and `kernel_raw!` in [`by_hand`], so that neither key compared is an
/// optimizer's. Beside the glyph, the same program with its three pieces as
/// three record parameters, summed by hand.
macro_rules! section_1_7 {
    ($expand:ident) => {
        $expand! {
            /// One oriented monotone arc piece.
            pub struct Row {
                pub x0: f32, pub e0x: f32, pub e1x: f32,
                pub y0: f32, pub e0y: f32, pub e1y: f32,
                pub sigma: f32, pub s: f32,
                pub lo: f32, pub hi: f32,
            }
            pub struct Bounds { pub x0: f32, pub y0: f32, pub x1: f32, pub y1: f32 }

            const PIXEL_CENTER: f32 = 0.5;
            const COVERAGE_SNAP: f32 = 1.0 / 1024.0;
            const NEARLY_ONE: f32 = 1.0 - COVERAGE_SNAP;

            fn coverage(f: f32) -> f32 {
                let c = f.abs().min(1.0);
                if c >= NEARLY_ONE { 1.0 } else if c <= COVERAGE_SNAP { 0.0 } else { c }
            }

            fn indicator(m: bool) -> f32 { if m { 1.0 } else { 0.0 } }

            /// χ: the region left of the arc, within its band.
            fn left_of_the_arc(p: Row, x: f32, y: f32) -> f32 {
                let b = p.e0y.max(0.0);
                let bx = p.e0x.max(0.0);
                let a = p.e1y.max(0.0) - b;
                let ax = p.e1x.max(0.0) - bx;
                let t = monotone_root(y - p.y0, b, a);
                let x_at_t = p.x0 + t * (bx + bx + ax * t);
                indicator(0.0 <= t) * indicator(t < 1.0) * indicator(x < x_at_t)
            }

            /// σ·∫∫χ over the pixel about (x, S·y), cut to the rows the piece
            /// reaches.
            fn piece_term(p: Row, x: f32, y: f32) -> f32 {
                let term = p.sigma * area(|u, v| left_of_the_arc(p, x + u, p.s * y + v));
                if (y > p.lo) & (y < p.hi) { term } else { 0.0 }
            }

            fn inside(b: Bounds, x: f32, y: f32) -> bool {
                (x >= b.x0) & (x <= b.x1) & (y >= b.y0) & (y <= b.y1)
            }

            /// The glyph with N pieces. Texel (i, j) holds coverage at
            /// (i+½, j+½). The pieces and the box are uniforms; N is the
            /// program.
            pub fn glyph<const N: usize>(pieces: [Row; N], bounds: Bounds) -> f32 {
                let (x, y) = (X + PIXEL_CENTER, Y + PIXEL_CENTER);
                let f: f32 = pieces.into_iter().map(|p| piece_term(p, x, y)).sum();
                if inside(bounds, x, y) { coverage(f) } else { 0.0 }
            }

            /// The glyph at `N = 3`, its pieces three record parameters
            /// summed by hand.
            pub fn three_pieces(p0: Row, p1: Row, p2: Row, bounds: Bounds) -> f32 {
                let (x, y) = (X + PIXEL_CENTER, Y + PIXEL_CENTER);
                let f: f32 = piece_term(p0, x, y) + piece_term(p1, x, y) + piece_term(p2, x, y);
                if inside(bounds, x, y) { coverage(f) } else { 0.0 }
            }
        }
    };
}

section_1_7!(kernel);

/// §1.7's block through `kernel_raw!`: the lowered shape, as written. Only
/// its programs are compared, so its `Args` records go unused.
#[allow(dead_code)]
mod by_hand {
    use pixelflow_compiler::kernel_raw;
    section_1_7!(kernel_raw);
}

/// The lattice the glyphs are baked over.
const FRAME: (usize, usize) = (9, 8);

/// The box: all of the lattice.
const EVERYWHERE: Bounds = Bounds {
    x0: 0.0,
    y0: 0.0,
    x1: 9.0,
    y1: 8.0,
};

/// A vertical side of a box at `x`, from `from` to `to` in `y`, as
/// `fonts/loop_blinn.rs`'s `Piece::oriented` and `piece_row` lay a line
/// out: its control point at its midpoint, reflected in `y` (`s = −1`) when
/// it runs down, so both of its steps rise, and `σ` the winding a ray to
/// `+X` picks up crossing it. Its band is its rows widened by half a pixel.
fn side(x: f32, from: f32, to: f32) -> Row {
    let s = if to < from { -1.0 } else { 1.0 };
    let half_step = s * (to - from) / 2.0;
    Row {
        x0: x,
        e0x: 0.0,
        e1x: 0.0,
        y0: s * from,
        e0y: half_step,
        e1y: half_step,
        sigma: s,
        s,
        lo: from.min(to) - 0.5,
        hi: from.max(to) + 0.5,
    }
}

/// The square `[2, 6) × [2, 6)` as three pieces: its right side rising, and
/// its left side falling, in two halves.
fn square() -> [Row; 3] {
    [
        side(6.0, 2.0, 6.0),
        side(2.0, 6.0, 4.0),
        side(2.0, 4.0, 2.0),
    ]
}

/// The rectangle `[1, 7) × [3, 5)`, as three pieces too.
fn wide() -> [Row; 3] {
    [
        side(7.0, 3.0, 5.0),
        side(1.0, 5.0, 4.0),
        side(1.0, 4.0, 3.0),
    ]
}

/// The key the JIT cache compiles under: the canonical form's shape bytes,
/// which number a kernel's arguments by slot and hold neither their
/// identities nor their values.
fn program_key(k: &Kernel) -> Vec<u8> {
    let (arena, root) = k.parts();
    canonical(arena, root).key
}

/// Every fold reachable from `root`.
fn folds(arena: &ExprArena, root: ExprId) -> Vec<Fold> {
    let mut seen = vec![false; arena.len()];
    let mut stack = vec![root];
    let mut out = Vec::new();
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        if let ExprNode::Reduce { fold, .. } = arena.node(id) {
            out.push(fold);
        }
        stack.extend(arena.children(id));
    }
    out
}

/// A texel's coverage of the square `[lo, hi)²`: the pixel `[i, i+1) ×
/// [j, j+1)` is inside it or out, since its edges are on the lattice's
/// integers.
fn square_coverage(index: usize, lo: usize, hi: usize) -> f32 {
    let (i, j) = (index % FRAME.0, index / FRAME.0);
    let inside = (lo..hi).contains(&i) && (lo..hi).contains(&j);
    if inside { 1.0 } else { 0.0 }
}

/// §1.7's glyph expands and bakes at `N = 0`, `1` and `3`, and its
/// arguments are each piece's ten fields, element-major, then the box's
/// four: `10·N + 4` uniforms, in declaration order.
#[test]
fn the_glyph_bakes_at_zero_one_and_three_pieces() {
    let lattice = Lattice::frame(FRAME.0, FRAME.1);

    let empty = glyph::<0>([], EVERYWHERE);
    assert_eq!(empty.uniforms().len(), 4, "the box's four");
    assert!(
        lattice.bake(&empty).buffer().iter().all(|&t| t == 0.0),
        "no pieces: the sum's identity, no ink"
    );

    let one = glyph::<1>([side(6.0, 2.0, 6.0)], EVERYWHERE);
    assert_eq!(one.uniforms().len(), 14);
    let baked = lattice.bake(&one);
    for (index, &texel) in baked.buffer().iter().enumerate() {
        let (i, j) = (index % FRAME.0, index / FRAME.0);
        let want = if i < 6 && (2..6).contains(&j) {
            1.0
        } else {
            0.0
        };
        assert_eq!(
            texel, want,
            "one side: everything left of it, in its rows, at ({i}, {j})"
        );
    }

    let pieces = square();
    let three = glyph::<3>(pieces, EVERYWHERE);
    let defaults: Vec<f32> = three.uniforms().iter().map(|u| u.default).collect();
    let fields = |p: &Row| {
        [
            p.x0, p.e0x, p.e1x, p.y0, p.e0y, p.e1y, p.sigma, p.s, p.lo, p.hi,
        ]
    };
    let declared: Vec<f32> = pieces
        .iter()
        .flat_map(fields)
        .chain([EVERYWHERE.x0, EVERYWHERE.y0, EVERYWHERE.x1, EVERYWHERE.y1])
        .collect();
    assert_eq!(defaults, declared, "element-major, then the box");
    let baked = lattice.bake(&three);
    for (index, &texel) in baked.buffer().iter().enumerate() {
        assert_eq!(texel, square_coverage(index, 2, 6), "texel {index}");
    }
}

/// A family is its copies: `glyph::<3>` is the program with three record
/// parameters summed by hand, `piece_term(p0, x, y) + piece_term(p1, x, y) +
/// piece_term(p2, x, y)` — one canonical key, and the same value in each
/// uniform slot. It holds the six integrals of its three `area`s and no
/// other fold: the family left no fold, binder or index behind.
#[test]
fn a_family_at_three_is_three_pieces_summed_by_hand() {
    let [p0, p1, p2] = square();
    let family = glyph::<3>([p0, p1, p2], EVERYWHERE);
    let as_raw = |p: Row| by_hand::Row {
        x0: p.x0,
        e0x: p.e0x,
        e1x: p.e1x,
        y0: p.y0,
        e0y: p.e0y,
        e1y: p.e1y,
        sigma: p.sigma,
        s: p.s,
        lo: p.lo,
        hi: p.hi,
    };
    let bounds = by_hand::Bounds {
        x0: EVERYWHERE.x0,
        y0: EVERYWHERE.y0,
        x1: EVERYWHERE.x1,
        y1: EVERYWHERE.y1,
    };
    let summed = by_hand::three_pieces(as_raw(p0), as_raw(p1), as_raw(p2), bounds);
    let (family_arena, family_root) = family.parts();
    let (summed_arena, summed_root) = summed.parts();
    let family_form = canonical(family_arena, family_root);
    let summed_form = canonical(summed_arena, summed_root);
    assert_eq!(
        family_form.key,
        summed_form.key,
        "family: {}\nsummed: {}",
        family_arena.display(family_root),
        summed_arena.display(summed_root)
    );
    let values = |form: &pixelflow_ir::key::Canonical| -> Vec<f32> {
        form.uniforms.iter().map(|u| u.default).collect()
    };
    assert_eq!(values(&family_form), values(&summed_form));

    let found = folds(family_arena, family_root);
    assert_eq!(found.len(), 6, "two integrals per piece: {found:?}");
    assert!(
        found.iter().all(|fold| matches!(fold, Fold::Interval(_))),
        "an area's integrals, and no fold for the family: {found:?}"
    );

    let the_family_through_kernel_raw = by_hand::glyph::<3>(
        [as_raw(p0), as_raw(p1), as_raw(p2)],
        by_hand::Bounds {
            x0: EVERYWHERE.x0,
            y0: EVERYWHERE.y0,
            x1: EVERYWHERE.x1,
            y1: EVERYWHERE.y1,
        },
    );
    assert_eq!(
        program_key(&the_family_through_kernel_raw),
        program_key(&family),
        "a template is emitted as lowered by either macro"
    );
}

/// The count is structural, and the pieces are uniforms: two glyphs of one
/// `N` are one program whatever their pieces, and `N = 3` and `N = 4` are
/// two.
#[test]
fn a_glyph_is_the_program_for_its_count() {
    let square = glyph::<3>(square(), EVERYWHERE);
    let wide = glyph::<3>(wide(), EVERYWHERE);
    assert_eq!(program_key(&square), program_key(&wide));
    let [a, b, c] = self::square();
    let four = glyph::<4>([a, b, c, side(8.0, 0.0, 1.0)], EVERYWHERE);
    assert_ne!(program_key(&square), program_key(&four));
    let lattice = Lattice::frame(FRAME.0, FRAME.1);
    assert_eq!(
        Manifold::compile(&square, lattice.extent)
            .code_bytes()
            .as_ptr(),
        Manifold::compile(&wide, lattice.extent)
            .code_bytes()
            .as_ptr(),
        "one compiled region"
    );
}

/// A program compiled once from `glyph::<3>`, rebound from `GlyphArgs` —
/// its family a field, its `N` inferred from the array — gives the pixels
/// of baking each call.
#[test]
fn a_compiled_glyph_is_rebound_from_its_args() {
    let lattice = Lattice::frame(FRAME.0, FRAME.1);
    let program = Manifold::compile(&glyph::<3>(square(), EVERYWHERE), lattice.extent);
    let mut block = program.block();
    let calls = [
        GlyphArgs {
            pieces: wide(),
            bounds: EVERYWHERE,
        },
        GlyphArgs {
            pieces: square(),
            bounds: Bounds {
                x0: 0.0,
                y0: 0.0,
                x1: 4.0,
                y1: 8.0,
            },
        },
    ];
    for args in calls {
        args.write_into(&mut block).expect("glyph's own arguments");
        let rebound = lattice.collapse(&program.bind(&[]).with_uniforms(&block));
        let baked = lattice.bake(&glyph::<3>(args.pieces, args.bounds));
        assert_eq!(rebound.buffer(), baked.buffer(), "{args:?}");
    }
}

/// The count an `Args` record's family has is the count a program was
/// compiled at, or `write_into` refuses — before it writes anything, since
/// the stream knows its length.
#[test]
fn args_of_another_count_are_refused_and_write_nothing() {
    let program = Manifold::compile(&glyph::<3>(square(), EVERYWHERE), [1, 1]);
    let mut block = program.block();
    let [a, b, _] = square();
    let two = GlyphArgs {
        pieces: [a, b],
        bounds: EVERYWHERE,
    };
    assert_eq!(
        two.write_into(&mut block),
        Err(ArityMismatch {
            declared: 34,
            supplied: 24
        })
    );
    assert_eq!(block.values(), program.block().values(), "nothing written");
}

kernel_raw! {
    /// A point with a weight.
    pub struct Mass { pub x: f32, pub w: f32 }

    /// Each point's weight where `X` is past it: an iteration nested in
    /// another over the same family is `N²` copies, each pair's.
    pub fn pairs<const N: usize>(masses: [Mass; N]) -> f32 {
        masses
            .into_iter()
            .map(|p| masses.into_iter().map(|q| p.w * q.x + X).sum::<f32>())
            .sum()
    }

    /// The same at `N = 2`, written out.
    pub fn pairs_of_two(a: Mass, b: Mass) -> f32 {
        ((a.w * a.x + X) + (a.w * b.x + X)) + ((b.w * a.x + X) + (b.w * b.x + X))
    }

    /// Each point plus its products with every point and `r`: the outer
    /// body more than the inner iteration, and `r` read by the inner body
    /// alone, so the outer copies reach it only through the inner ones.
    pub fn deep<const N: usize>(a: [f32; N], r: f32) -> f32 {
        a.into_iter()
            .map(|p| p + a.into_iter().map(|q| p * q + r).sum::<f32>())
            .sum()
    }

    /// The same at `N = 2`, written out.
    pub fn deep_of_two(a0: f32, a1: f32, r: f32) -> f32 {
        (a0 + ((a0 * a0 + r) + (a0 * a1 + r))) + (a1 + ((a1 * a0 + r) + (a1 * a1 + r)))
    }

    /// Each point's share of the total weight at `X`: an iteration reading
    /// another's result, which every one of its copies shares.
    pub fn shares<const N: usize>(masses: [Mass; N]) -> f32 {
        let total: f32 = masses.into_iter().map(|m| m.w * X).sum();
        masses.into_iter().map(|m| m.x * total).sum()
    }

    /// The same at `N = 2`, written out.
    pub fn shares_of_two(a: Mass, b: Mass) -> f32 {
        let total = a.w * X + b.w * X;
        a.x * total + b.x * total
    }

    /// Every triple's `a.x·b.w + c.x·r`: an iteration nested two deep over
    /// one family, so the innermost template's copies are made inside the
    /// middle one's, whose copies are made inside the outer one's, and each
    /// reads its elements at the program's slots.
    pub fn triples<const N: usize>(masses: [Mass; N], r: f32) -> f32 {
        masses
            .into_iter()
            .map(|a| {
                masses
                    .into_iter()
                    .map(|b| masses.into_iter().map(|c| a.x * b.w + c.x * r).sum::<f32>())
                    .sum::<f32>()
            })
            .sum()
    }

    /// The same at `N = 2`, written out: its eight terms.
    pub fn triples_of_two(p: Mass, q: Mass, r: f32) -> f32 {
        (((p.x * p.w + p.x * r) + (p.x * p.w + q.x * r))
            + ((p.x * q.w + p.x * r) + (p.x * q.w + q.x * r)))
            + (((q.x * p.w + p.x * r) + (q.x * p.w + q.x * r))
                + ((q.x * q.w + p.x * r) + (q.x * q.w + q.x * r)))
    }

    /// Three families nested three deep inside a fold, the innermost body
    /// reading every element, the fold's index, an argument and a record's
    /// field; each level's body more than the level inside it.
    pub fn layers<const A: usize, const B: usize, const C: usize>(
        r: f32,
        outer: [Mass; A],
        middle: [f32; B],
        inner: [Mass; C],
        m: Mass,
    ) -> f32 {
        (0..2)
            .map(|i| {
                outer
                    .into_iter()
                    .map(|p| {
                        middle
                            .into_iter()
                            .map(|q| {
                                inner
                                    .into_iter()
                                    .map(|s| p.x * q + s.w * (i as f32) + m.x * r)
                                    .sum::<f32>()
                                    * q
                            })
                            .sum::<f32>()
                            + p.w
                    })
                    .sum::<f32>()
            })
            .sum()
    }

    /// Two masses' slots as one record, element-major, as a family of two
    /// declares them.
    pub struct TwoMasses { pub x0: f32, pub w0: f32, pub x1: f32, pub w1: f32 }

    /// Two scalars' slots as one record.
    pub struct TwoScalars { pub q0: f32, pub q1: f32 }

    /// The same with two elements in each family, written out, each
    /// family's slots a record of their own.
    pub fn layers_of_two(r: f32, p: TwoMasses, q: TwoScalars, s: TwoMasses, m: Mass) -> f32 {
        (0..2)
            .map(|i| {
                ((((p.x0 * q.q0 + s.w0 * (i as f32) + m.x * r)
                    + (p.x0 * q.q0 + s.w1 * (i as f32) + m.x * r))
                    * q.q0
                    + ((p.x0 * q.q1 + s.w0 * (i as f32) + m.x * r)
                        + (p.x0 * q.q1 + s.w1 * (i as f32) + m.x * r))
                        * q.q1)
                    + p.w0)
                    + ((((p.x1 * q.q0 + s.w0 * (i as f32) + m.x * r)
                        + (p.x1 * q.q0 + s.w1 * (i as f32) + m.x * r))
                        * q.q0
                        + ((p.x1 * q.q1 + s.w0 * (i as f32) + m.x * r)
                            + (p.x1 * q.q1 + s.w1 * (i as f32) + m.x * r))
                            * q.q1)
                        + p.w1)
            })
            .sum()
    }
}

/// An iteration nested in another over the same family is `N²` copies:
/// `Σ_p Σ_q (p.w·q.x + X)` at `N = 2` is its four terms written out.
#[test]
fn a_nested_iteration_is_every_pair() {
    let (a, b) = (Mass { x: 1.0, w: 2.0 }, Mass { x: 3.0, w: 5.0 });
    let nested = pairs::<2>([a, b]);
    assert_eq!(program_key(&nested), program_key(&pairs_of_two(a, b)));
    assert_eq!(
        Lattice::eval_at(&nested, 0.5, 0.0),
        (2.0 * 1.0 + 0.5) + (2.0 * 3.0 + 0.5) + (5.0 * 1.0 + 0.5) + (5.0 * 3.0 + 0.5)
    );
}

/// A nested iteration's body may read what its enclosing body does not: `r`
/// reaches each outer copy through its inner copies, and the program is the
/// terms written out.
#[test]
fn a_nested_iteration_reads_what_its_enclosing_body_does_not() {
    let nested = deep::<2>([1.0, 2.0], 3.0);
    assert_eq!(
        program_key(&nested),
        program_key(&deep_of_two(1.0, 2.0, 3.0))
    );
    // 1 + (1·1 + 3) + (1·2 + 3), and 2 + (2·1 + 3) + (2·2 + 3).
    assert_eq!(Lattice::eval_at(&nested, 0.5, 0.0), 10.0 + 14.0);
    assert_eq!(Lattice::eval_at(&deep::<0>([], 3.0), 0.5, 0.0), 0.0);
}

/// An iteration reading another's result is its copies each over that one
/// result: the program is the terms written out, the total built once.
#[test]
fn an_iteration_reading_another_shares_its_result() {
    let (a, b) = (Mass { x: 1.0, w: 2.0 }, Mass { x: 3.0, w: 5.0 });
    let shared = shares::<2>([a, b]);
    assert_eq!(program_key(&shared), program_key(&shares_of_two(a, b)));
    // X = 0.5: the total is 2·½ + 5·½, and 1·3.5 + 3·3.5.
    assert_eq!(Lattice::eval_at(&shared, 0.5, 0.0), 14.0);
}

/// `family` is `written_out`: one canonical key, and the same value in each
/// uniform slot — so every copy reads its elements from the slots the host
/// function declared them in, not merely slots of the same shape.
fn assert_is_written_out(family: &Kernel, written_out: &Kernel) {
    let (family_arena, family_root) = family.parts();
    let (written_arena, written_root) = written_out.parts();
    let family_form = canonical(family_arena, family_root);
    let written_form = canonical(written_arena, written_root);
    assert_eq!(
        family_form.key,
        written_form.key,
        "family: {}\nwritten out: {}",
        family_arena.display(family_root),
        written_arena.display(written_root)
    );
    let values = |form: &pixelflow_ir::key::Canonical| -> Vec<u32> {
        form.uniforms.iter().map(|u| u.default.to_bits()).collect()
    };
    assert_eq!(values(&family_form), values(&written_form));
}

/// An iteration nested two deep over one family is `N³` copies: at `N = 2`,
/// `Σ_a Σ_b Σ_c (a.x·b.w + c.x·r)` is its eight terms written out. The
/// innermost copies are made inside the middle iteration's template, whose
/// copies are made inside the outer one's, so each reads its element where
/// the host function declared it only if every template on the way holds
/// the slots of the arena it is copied into. Counting one arena's table
/// while copying another's passes every iteration one deep, and fails this.
#[test]
fn an_iteration_nested_two_deep_is_every_triple() {
    let (p, q) = (Mass { x: 1.0, w: 2.0 }, Mass { x: 3.0, w: 0.25 });
    let r = 0.5;
    let nested = triples::<2>([p, q], r);
    assert_is_written_out(&nested, &triples_of_two(p, q, r));
    let masses = [p, q];
    let by_rust: f32 = masses
        .iter()
        .map(|a| {
            masses
                .iter()
                .map(|b| masses.iter().map(|c| a.x * b.w + c.x * r).sum::<f32>())
                .sum::<f32>()
        })
        .sum();
    assert_eq!(Lattice::eval_at(&nested, 0.5, 0.0), by_rust);
    assert_eq!(Lattice::eval_at(&triples::<0>([], r), 0.5, 0.0), 0.0);
}

/// Three families nested three deep inside a fold are their copies: with
/// two elements each, the program is the fold over its terms written out,
/// every level reading the one outside it, the index, an argument and a
/// record's field.
#[test]
fn three_families_nested_in_a_fold_are_their_terms_written_out() {
    let (p0, p1) = (Mass { x: 1.0, w: 2.0 }, Mass { x: 3.0, w: 0.25 });
    let (q0, q1) = (0.5, 1.25);
    let (s0, s1) = (Mass { x: 5.0, w: 0.75 }, Mass { x: 7.0, w: 1.5 });
    let (r, m) = (0.5, Mass { x: 0.125, w: 9.0 });
    let nested = layers::<2, 2, 2>(r, [p0, p1], [q0, q1], [s0, s1], m);
    let two = |a: Mass, b: Mass| TwoMasses {
        x0: a.x,
        w0: a.w,
        x1: b.x,
        w1: b.w,
    };
    let written_out = layers_of_two(r, two(p0, p1), TwoScalars { q0, q1 }, two(s0, s1), m);
    assert_is_written_out(&nested, &written_out);
    let by_rust: f32 = (0..2)
        .map(|i| {
            [p0, p1]
                .iter()
                .map(|p| {
                    [q0, q1]
                        .iter()
                        .map(|q| {
                            [s0, s1]
                                .iter()
                                .map(|s| p.x * q + s.w * (i as f32) + m.x * r)
                                .sum::<f32>()
                                * q
                        })
                        .sum::<f32>()
                        + p.w
                })
                .sum::<f32>()
        })
        .sum();
    assert_eq!(Lattice::eval_at(&nested, 0.5, 0.0), by_rust);
}

/// A family is declared, and streamed by `write_into`, where it is written
/// among the entry's parameters: after a value, before a record. Each slot
/// weighs a different power of ten, so a value in another's slot is another
/// number.
mod declaration_order {
    use pixelflow_compiler::kernel;

    kernel! {
        /// Two scalars.
        pub struct Pair { pub x: f32, pub y: f32 }
        /// A box.
        pub struct Bounds { pub x0: f32, pub y0: f32, pub x1: f32, pub y1: f32 }

        /// A family between a value and a record.
        pub fn between<const N: usize>(a: f32, pieces: [Pair; N], b: Bounds) -> f32 {
            a * 100.0 + pieces.into_iter().map(|p| p.x * 10.0 + p.y).sum::<f32>()
                + b.x0 * 1000.0
                + b.y1 * X
        }
    }
}

/// A family between a value and a record is declared between them, element
/// by element, and a compiled program rebound from the entry's `Args` gives
/// that call's pixels; so for the closure form's `f32`s, rebound by
/// position.
#[test]
fn a_family_between_other_parameters_is_declared_and_streamed_where_it_is_written() {
    use declaration_order::{BetweenArgs, Bounds, Pair, between};
    let lattice = Lattice::frame(4, 2);

    let first = between::<2>(
        2.0,
        [Pair { x: 3.0, y: 5.0 }, Pair { x: 7.0, y: 11.0 }],
        Bounds {
            x0: 13.0,
            y0: 17.0,
            x1: 19.0,
            y1: 23.0,
        },
    );
    let defaults: Vec<f32> = first.uniforms().iter().map(|u| u.default).collect();
    assert_eq!(
        defaults,
        [2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0],
        "`a`, then the pairs element-major, then the box"
    );
    let program = Manifold::compile(&first, lattice.extent);
    let mut block = program.block();
    let args = BetweenArgs {
        a: 0.5,
        pieces: [Pair { x: 0.25, y: 4.0 }, Pair { x: -1.0, y: 0.75 }],
        b: Bounds {
            x0: 0.125,
            y0: 9.0,
            x1: 8.0,
            y1: 1.5,
        },
    };
    args.write_into(&mut block)
        .expect("between's own arguments");
    let rebound = lattice.collapse(&program.bind(&[]).with_uniforms(&block));
    let baked = lattice.bake(&between(args.a, args.pieces, args.b));
    assert_eq!(rebound.buffer(), baked.buffer());

    let closure = kernel!(|w: f32, v: [f32; 3], z: f32| w * 100.0
        + v.into_iter().map(|e| e * 10.0).sum::<f32>()
        + z * X);
    let defaults: Vec<f32> = closure(2.0, [3.0, 5.0, 7.0], 11.0)
        .uniforms()
        .iter()
        .map(|u| u.default)
        .collect();
    assert_eq!(defaults, [2.0, 3.0, 5.0, 7.0, 11.0]);
    let program = Manifold::compile(&closure(2.0, [3.0, 5.0, 7.0], 11.0), lattice.extent);
    let mut block = program.block();
    block
        .set_declared([0.5, 0.25, -1.0, 4.0, 1.5])
        .expect("the closure's own arity");
    let rebound = lattice.collapse(&program.bind(&[]).with_uniforms(&block));
    let baked = lattice.bake(&closure(0.5, [0.25, -1.0, 4.0], 1.5));
    assert_eq!(rebound.buffer(), baked.buffer());
}

/// The macro doc's family example, compiled, so the doc cannot drift from
/// the language: in a module of its own, since its record is its block's.
mod the_macro_docs_example {
    use pixelflow_compiler::kernel;

    kernel! {
        /// A disc: its centre and its radius.
        pub struct Disc { pub cx: f32, pub cy: f32, pub r: f32 }

        fn inside(d: Disc, x: f32, y: f32) -> f32 {
            let (dx, dy) = (x - d.cx, y - d.cy);
            if dx * dx + dy * dy < d.r * d.r { 1.0 } else { 0.0 }
        }

        /// How many of the discs cover the sample: `N` is the program, and
        /// the discs are its uniforms.
        pub fn cover<const N: usize>(discs: [Disc; N]) -> f32 {
            discs.into_iter().map(|d| inside(d, X, Y)).sum()
        }
    }
}

/// The macro doc's family example bakes each disc's cover, and the program
/// compiled from one call, rebound from `CoverArgs`, gives another call's
/// pixels.
#[test]
fn the_macro_docs_family_example_bakes_and_rebinds() {
    use the_macro_docs_example::{CoverArgs, Disc, cover};
    let lattice = Lattice::frame(8, 8);
    let a = Disc {
        cx: 3.0,
        cy: 3.0,
        r: 2.0,
    };
    let b = Disc {
        cx: 5.0,
        cy: 3.0,
        r: 2.0,
    };
    let once = lattice.bake(&cover([a, b]));
    assert_eq!(Lattice::eval_at(&cover([a, b]), 4.0, 3.0), 2.0, "in both");
    assert_eq!(Lattice::eval_at(&cover([a, b]), 2.0, 3.0), 1.0, "in `a`");
    assert!(once.buffer().contains(&2.0), "some texel is in both");

    let program = Manifold::compile(&cover([a, b]), lattice.extent);
    let mut block = program.block();
    let args = CoverArgs {
        discs: [b, Disc { r: 3.0, ..a }],
    };
    args.write_into(&mut block).expect("cover's own arguments");
    let again = lattice.collapse(&program.bind(&[]).with_uniforms(&block));
    assert_eq!(again.buffer(), lattice.bake(&cover(args.discs)).buffer());
}

/// A block written where no prelude is in scope: records, a `pub const`, a
/// helper, a tuple `let`, a family iterated two deep in a fold, a family of
/// `f32`s, their `Args` stream, and the closure form's family. The
/// expansion names every item by path, so it expands in a
/// `#[no_implicit_prelude]` module as anywhere else — which a bare `Some`
/// or an `Iterator` method called through the prelude's trait would not.
#[no_implicit_prelude]
mod without_a_prelude {
    ::pixelflow_compiler::kernel! {
        /// A point with a weight.
        pub struct Mass { pub x: f32, pub w: f32 }

        /// Half.
        pub const HALF: f32 = 0.5;

        fn weighed(m: Mass, r: f32) -> f32 { m.x * r + m.w }

        /// Every pair's weighed product, plus each fold index, plus a half
        /// where some `v` is past `X`.
        pub fn nested<const N: usize>(masses: [Mass; N], v: [f32; 2], r: f32) -> f32 {
            let (a, b) = (r, X);
            (0..2)
                .map(|i| {
                    masses
                        .into_iter()
                        .map(|p| {
                            masses
                                .into_iter()
                                .map(|q| weighed(p, a) * q.w + (i as f32))
                                .sum::<f32>()
                        })
                        .sum::<f32>()
                })
                .sum::<f32>()
                + if v.into_iter().any(|e| b < e) { HALF } else { 0.0 }
        }
    }

    /// The closure form's family, scaled.
    pub fn scaled(v: [f32; 2], r: f32) -> ::pixelflow_core::Kernel {
        let scaled = ::pixelflow_compiler::kernel!(|v: [f32; 2], r: f32| v
            .into_iter()
            .map(|e| e * r)
            .sum::<f32>());
        scaled(v, r)
    }
}

/// The block without a prelude means what it says, and its program rebinds
/// from its `Args`.
#[test]
fn a_family_expands_where_no_prelude_is_in_scope() {
    use without_a_prelude::{HALF, Mass, NestedArgs, nested, scaled};
    let masses = [Mass { x: 1.0, w: 2.0 }, Mass { x: 3.0, w: 0.25 }];
    let (v, r) = ([0.25, 4.0], 0.5);
    let x = 0.5;
    let by_rust: f32 = (0..2)
        .map(|i| {
            masses
                .iter()
                .map(|p| {
                    masses
                        .iter()
                        .map(|q| (p.x * r + p.w) * q.w + i as f32)
                        .sum::<f32>()
                })
                .sum::<f32>()
        })
        .sum::<f32>()
        + if v.iter().any(|&e| x < e) { HALF } else { 0.0 };
    let kernel = nested::<2>(masses, v, r);
    assert_eq!(Lattice::eval_at(&kernel, x, 0.0), by_rust);

    let lattice = Lattice::frame(4, 2);
    let program = Manifold::compile(&kernel, lattice.extent);
    let mut block = program.block();
    let args = NestedArgs {
        masses: [Mass { x: 0.5, w: 1.5 }, Mass { x: -1.0, w: 4.0 }],
        v: [3.0, 0.125],
        r: 2.0,
    };
    args.write_into(&mut block).expect("nested's own arguments");
    let rebound = lattice.collapse(&program.bind(&[]).with_uniforms(&block));
    let baked = lattice.bake(&nested(args.masses, args.v, args.r));
    assert_eq!(rebound.buffer(), baked.buffer());

    assert_eq!(Lattice::eval_at(&scaled([1.0, 2.0], 0.5), 0.0, 0.0), 1.5);
}
