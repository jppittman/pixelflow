// §1.7's block (docs/plans/2026-09-25-the-language-is-kernel.md): a font's
// glyph, in the language. One piece of a glyph is an entry over a record of
// ten `f32`s, each a uniform; a glyph is its pieces' terms summed under its
// box. The language has no collection types, so the sum is composition: the
// host walks a glyph's pieces and sums their `one_piece` instances into one
// ink with `sum2`, a kernel-typed entry, and `glyph` takes that ink as a
// kernel-typed argument (Phase D-a).
//
// Written once and included by `pixelflow-graphics`'s `fonts/loop_blinn`
// tests (`kernel_copy.rs`), which pin `one_piece` against the builder's
// `piece_term` (`fonts/loop_blinn.rs`) by canonical key and uniform values,
// and real glyphs composed this way against the builder's `glyph` by their
// pixels, until Phase C of the plan deletes the builder.
//
// Included with `include!`, so it holds only the macro: `section_1_7!(kernel)`
// expands the block.

/// §1.7's block, expanded by `$expand` — `kernel` or `kernel_raw`.
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

            /// A glyph's box: the outline's bounding box, four uniforms.
            /// Coverage reaches half a pixel past it, which `inside`
            /// reaches too, so the host passes the outline's own box.
            pub struct Bounds { pub x0: f32, pub y0: f32, pub x1: f32, pub y1: f32 }

            const PIXEL_CENTER: f32 = 0.5;
            const COVERAGE_SNAP: f32 = 1.0 / 1024.0;
            const NEARLY_ONE: f32 = 1.0 - COVERAGE_SNAP;
            const PIXEL_HALF: f32 = 0.5;
            const ONE_THIRD: f32 = 1.0 / 3.0;
            const ROOT_FLOOR: f32 = 1.0 / 1_267_650_600_228_229_401_496_703_205_376.0;

            /// `τ(δ) = δ / max(step + √max(step² + bend·δ, 0), ROOT_FLOOR)`:
            /// the parameter at which the rise `t·(2·step + bend·t)` reaches
            /// the height `δ`, the reciprocal exact —
            /// `fonts/loop_blinn.rs`'s `Rise::monotone_root`, which carries
            /// its law.
            fn monotone_root(delta: f32, step: f32, bend: f32) -> f32 {
                delta * (1.0 / (step + (step * step + bend * delta).max(0.0).sqrt()).max(ROOT_FLOOR))
            }

            /// `2·step + bend·s`: a rise `q(t) = t·(2·step + bend·t)` is
            /// `t` times it at `s = t`, and climbs `(t − s)` times it at
            /// `s + t` from `s` to `t`.
            fn slope_through(step: f32, bend: f32, s: f32) -> f32 {
                step + step + bend * s
            }

            /// The area of the pixel about `(x, y)` left of the arc, within
            /// its band, in closed form: `fonts/loop_blinn.rs`'s
            /// `RisingArc::pixel_area`, which carries the derivation.
            fn piece_area(p: Row, x: f32, y: f32) -> f32 {
                let b = p.e0y.max(0.0);
                let bx = p.e0x.max(0.0);
                let a = p.e1y.max(0.0) - b;
                let ax = p.e1x.max(0.0) - bx;
                let across = x - p.x0;
                let up = y - p.y0;
                let left = across - PIXEL_HALF;
                let right = across + PIXEL_HALF;
                // Where the arc enters and leaves the pixel's rows, and
                // where it reaches the pixel's left and right edges.
                let t0 = monotone_root(up - PIXEL_HALF, b, a).clamp(0.0, 1.0);
                let t1 = monotone_root(up + PIXEL_HALF, b, a).clamp(0.0, 1.0);
                let t_left = monotone_root(left, bx, ax).clamp(t0, t1);
                let t_right = monotone_root(right, bx, ax).clamp(t0, t1);
                let right_of_the_pixel = (t1 - t_right) * slope_through(b, a, t_right + t1);
                let x_left = t_left * slope_through(bx, ax, t_left) - left;
                let x_right = t_right * slope_through(bx, ax, t_right) - left;
                let rise = (t_right - t_left) * slope_through(b, a, t_left + t_right);
                let trapezoid = PIXEL_HALF * (x_left + x_right) * rise;
                let w = t_right - t_left;
                let bow = ONE_THIRD * (bx * a - b * ax) * (w * w * w);
                right_of_the_pixel + (trapezoid + bow)
            }

            /// σ·A over the pixel about (x, S·y), cut to the rows the piece
            /// reaches.
            fn piece_term(p: Row, x: f32, y: f32) -> f32 {
                let term = p.sigma * piece_area(p, x, p.s * y);
                if (y > p.lo) & (y < p.hi) { term } else { 0.0 }
            }

            /// Coverage, from a signed area: `min(|f|, 1)`, its ends snapped.
            fn coverage(f: f32) -> f32 {
                let c = f.abs().min(1.0);
                if c >= NEARLY_ONE { 1.0 } else if c <= COVERAGE_SNAP { 0.0 } else { c }
            }

            /// Whether the pixel about (x, y) can meet the outline: its
            /// centre within half a pixel of the outline's box. Every
            /// piece's term is exactly 0 farther out.
            fn inside(b: Bounds, x: f32, y: f32) -> bool {
                (x >= b.x0 - PIXEL_HALF) & (x <= b.x1 + PIXEL_HALF)
                    & (y >= b.y0 - PIXEL_HALF) & (y <= b.y1 + PIXEL_HALF)
            }

            /// One piece's term at the sample, over its own ten uniforms.
            /// The host composes one instance per piece.
            pub fn one_piece(p: Row) -> f32 {
                piece_term(p, X, Y)
            }

            /// Two kernels summed at the sample: the operation of the
            /// monoid a glyph's ink is, which the host folds its pieces'
            /// instances with — a balanced tree, by index.
            pub fn sum2(a: impl Fn(f32, f32) -> f32, b: impl Fn(f32, f32) -> f32) -> f32 {
                a(X, Y) + b(X, Y)
            }

            /// One glyph. Texel (i, j) holds coverage at (i+½, j+½). `ink`
            /// is the sum of the glyph's pieces, composed by the host, and
            /// `ink(x, y)` reads it at the pixel's centre: application is
            /// contramap (§1.2).
            pub fn glyph(ink: impl Fn(f32, f32) -> f32, bounds: Bounds) -> f32 {
                let (x, y) = (X + PIXEL_CENTER, Y + PIXEL_CENTER);
                if inside(bounds, x, y) { coverage(ink(x, y)) } else { 0.0 }
            }
        }
    };
}
