//! Withheld real-shader benchmark set — ShaderToy / iquilezles.org ports for
//! the `FINAL` (publication-only) corpus tier.
//!
//! docs/plans/2026-08-05-egraph-nnue-research-workflow.md §0.2 defines
//! `FINAL` as "real pixelflow kernels (the 5 named production kernels + real
//! `kernel!` call sites harvested from the repo + validated ShaderToy
//! imports) + one synthetic family. Never used for any selection decision;
//! touched only for the paper's claimed numbers." The twelve kernels below
//! are that "validated ShaderToy imports" line item. They are registered in
//! `corpus_split.toml`'s `[final].kernels` exactly the way the five original
//! named kernels (`swirl`, `circle_sdf`, `poly`, `redundant`, `normalize`,
//! defined beside `gen_bench_corpus`'s `named_kernel`) are — same discipline:
//! **selection runs (the DEV tier, no `--final-eval`) never see these.**
//!
//! # What "port" means here
//!
//! A pixelflow kernel is pure arithmetic over coordinates X/Y/Z/W — no
//! texture sampling, no buffers/gather, no data-dependent loops. That rules
//! out transcribing a raymarch loop or a texture-backed hash/noise verbatim.
//! Every port below is therefore an honest *computational kernel* of the
//! cited shader's per-pixel math, not a pixel-perfect replica: raymarch loops
//! become direct closed-form field evaluations, texture noise becomes a
//! sin-based pseudo-hash, and iteration counts (fractals, fbm octaves) are
//! fixed and unrolled by hand. Each function's doc comment states exactly
//! what was kept and what was simplified.
//!
//! # Licensing
//!
//! ShaderToy's default license is CC BY-NC-SA 3.0; individual shaders may
//! declare others (iq marks many of his own MIT). ShaderToy's shader pages
//! sit behind bot-verification that blocks automated fetch (confirmed
//! 2026-08-27: both direct fetch and a text-extraction proxy returned the
//! "verifying you are not a bot" interstitial, never shader content) — so no
//! GLSL source was copied from ShaderToy for any kernel here, and no
//! per-shader license override could be positively confirmed except where
//! noted. Every entry below is attributed and, absent a confirmed override,
//! treated conservatively as ShaderToy's site default. iquilezles.org article
//! pages fetched cleanly and are cited directly; that site states no explicit
//! license, so formulas taken from it are likewise treated conservatively.
//!
//! | Name | Title | Author | Source | License | Category |
//! |---|---|---|---|---|---|
//! | `cosine_palette` | Cosine Color Palette (article) | Inigo Quilez (iq) | <https://iquilezles.org/articles/palettes/> (example: shadertoy.com/view/Xl2GRc) | CC BY-NC-SA 3.0 (default, unconfirmed) | transcendental-heavy, famous |
//! | `smooth_min_scene` | smin (smooth minimum) (article) | Inigo Quilez (iq) | <https://iquilezles.org/articles/smin/> (example: shadertoy.com/view/DlVcW1) | CC BY-NC-SA 3.0 (default, unconfirmed) | SDF-composition-heavy |
//! | `mandelbrot_distance` | distance to the Mandelbrot set (article) | Inigo Quilez (iq) | <https://iquilezles.org/articles/distancefractals/> (example: shadertoy.com/view/lsX3W4) | CC BY-NC-SA 3.0 (default, unconfirmed) | select/branch-heavy, famous |
//! | `star_sdf` | 2D distance functions — sdPentagram (article) | Inigo Quilez (iq) | <https://iquilezles.org/articles/distfunctions2d/> (example: shadertoy.com/view/t3X3z4) | CC BY-NC-SA 3.0 (default, unconfirmed) | SDF-composition-heavy |
//! | `gyroid_slice` | gyroid SDF | zzggbb | <https://www.shadertoy.com/view/wtfSRS> (2019-07-17) | CC BY-NC-SA 3.0 (default, unconfirmed) | SDF-composition-heavy, famous technique |
//! | `plasma` | Plasma 90x | bitek | <https://www.shadertoy.com/view/4ssGR7> (2013-04-16) | CC BY-NC-SA 3.0 (default, unconfirmed) | transcendental-heavy |
//! | `domain_warp_fbm` | Domain Warping (article) | Inigo Quilez (iq) | <https://iquilezles.org/articles/warp/> (example: shadertoy.com/view/4s23zzM) | CC BY-NC-SA 3.0 (default, unconfirmed) | transcendental-heavy, famous |
//! | `kaleidoscope_fold` | Kaleidoscope Tutorial | deliaev | <https://www.shadertoy.com/view/WdcSRr> (2020-07-22) | CC BY-NC-SA 3.0 (default, unconfirmed) | select/branch-heavy |
//! | `metaballs` | Metaball | unresolved handle | <https://www.shadertoy.com/view/Xdl3Wl> (2013); technique: Blinn 1982 | CC BY-NC-SA 3.0 (default, unconfirmed) | select/branch-heavy |
//! | `julia_set` | Julia - Distance 2 | Inigo Quilez (iq) | <https://www.shadertoy.com/view/3llyzl> | CC BY-NC-SA 3.0 (confirmed) | select/branch-heavy, famous |
//! | `smoothstep_vignette` | smoothstep (glossary) | Patricio Gonzalez Vivo (Book of Shaders) | <https://thebookofshaders.com/glossary/?search=smoothstep> | CC BY-NC-SA (site-wide, version unconfirmed) | fma-friendly polynomial |
//! | `torus_slice` | Signed Distance Functions — sdTorus (article) | Inigo Quilez (iq) | <https://iquilezles.org/articles/distfunctions/> (playlist: shadertoy.com/playlist/43cXRl) | CC BY-NC-SA 3.0 (default, unconfirmed) | SDF-composition-heavy |
//!
//! All twelve fetched/searched 2026-08-27.
//!
//! # What was tried and abandoned
//!
//! An earlier draft attempted a true multi-octave (3+) domain warp
//! (`f(p + 4*warp1(p + 4*warp2(p)))`, iq's exact two-level "Warping" recipe)
//! and a full raymarched "Seascape"-style ocean. Both were dropped: the
//! two-level warp's node count (each level re-evaluates a multi-octave fbm
//! twice) blew well past the corpus's ~30-400 node realistic band before
//! reaching iq's actual visual richness, and a raymarched ocean has no
//! honest "2D slice" simplification — collapsing the marched loop to one
//! step stops being recognizably the cited shader's computation rather than
//! a simplification of it. `domain_warp_fbm` below keeps one warp level and
//! two fbm octaves instead.

use pixelflow_ir::{ExprArena, ExprId, OpKind};
use std::f32::consts::TAU;

/// Minimal builder sugar over [`ExprArena`]'s `push_*` calls, so the ports
/// below read close to the cited GLSL instead of a wall of `push_binary`.
/// Every method is a direct one-line wrapper — no new semantics.
trait Build {
    fn k(&mut self, v: f32) -> ExprId;
    fn var(&mut self, i: u8) -> ExprId;
    fn add(&mut self, x: ExprId, y: ExprId) -> ExprId;
    fn sub(&mut self, x: ExprId, y: ExprId) -> ExprId;
    fn mul(&mut self, x: ExprId, y: ExprId) -> ExprId;
    fn div(&mut self, x: ExprId, y: ExprId) -> ExprId;
    fn sqrt(&mut self, x: ExprId) -> ExprId;
    fn abs(&mut self, x: ExprId) -> ExprId;
    fn floor(&mut self, x: ExprId) -> ExprId;
    fn minv(&mut self, x: ExprId, y: ExprId) -> ExprId;
    fn maxv(&mut self, x: ExprId, y: ExprId) -> ExprId;
    fn sin(&mut self, x: ExprId) -> ExprId;
    fn cos(&mut self, x: ExprId) -> ExprId;
    fn atan2(&mut self, y: ExprId, x: ExprId) -> ExprId;
    fn ln(&mut self, x: ExprId) -> ExprId;
    fn log2(&mut self, x: ExprId) -> ExprId;
    fn gt(&mut self, x: ExprId, y: ExprId) -> ExprId;
    fn ge(&mut self, x: ExprId, y: ExprId) -> ExprId;
    fn mul_add(&mut self, x: ExprId, y: ExprId, z: ExprId) -> ExprId;
    fn select(&mut self, cond: ExprId, t: ExprId, f: ExprId) -> ExprId;

    /// `x * x`.
    fn sq(&mut self, x: ExprId) -> ExprId {
        self.mul(x, x)
    }
    /// `max(min(v, hi), lo)`.
    fn clampf(&mut self, v: ExprId, lo: f32, hi: f32) -> ExprId {
        let lo = self.k(lo);
        let hi = self.k(hi);
        let h = self.minv(v, hi);
        self.maxv(h, lo)
    }
    /// Coordinate `i`, clamped into `[-half, half]` — the "scale coordinates
    /// into a sensible viewport" step every port below starts with, so a
    /// quarantine grid point at magnitude 1e4 still lands somewhere sane
    /// (CLAUDE.md "Floating point at the edges": stay inside `TRIG_DOMAIN`
    /// and avoid NaN-producing regions by construction, not by hoping the
    /// input is small).
    fn viewport(&mut self, i: u8, half: f32) -> ExprId {
        let v = self.var(i);
        self.clampf(v, -half, half)
    }
    /// A lattice-invariant scalar the caller supplies: the plane a 3-D
    /// shader is cut at, or a shader's clock. It was the Z coordinate until
    /// a lattice became two axes — the ports below always sampled it at one
    /// value per call, which is what a uniform is. Never folded, so the
    /// expression keeps its shape.
    fn arg(&mut self, default: f32) -> ExprId;
    /// `x*x + y*y`.
    fn length2(&mut self, x: ExprId, y: ExprId) -> ExprId {
        let xx = self.sq(x);
        let yy = self.sq(y);
        self.add(xx, yy)
    }
    /// `sqrt(x*x + y*y)` — always a non-negative radicand (sum of two
    /// squares), so `Sqrt` never sees a negative input here.
    fn length(&mut self, x: ExprId, y: ExprId) -> ExprId {
        let l2 = self.length2(x, y);
        self.sqrt(l2)
    }
}

impl Build for ExprArena {
    fn k(&mut self, v: f32) -> ExprId {
        self.push_const(v)
    }
    fn var(&mut self, i: u8) -> ExprId {
        self.push_var(i)
    }
    fn arg(&mut self, default: f32) -> ExprId {
        let slot = self.declare_uniform(pixelflow_ir::Uniform::new(default).decl());
        self.push_uniform(slot)
    }
    fn add(&mut self, x: ExprId, y: ExprId) -> ExprId {
        self.push_binary(OpKind::Add, x, y)
    }
    fn sub(&mut self, x: ExprId, y: ExprId) -> ExprId {
        self.push_binary(OpKind::Sub, x, y)
    }
    fn mul(&mut self, x: ExprId, y: ExprId) -> ExprId {
        self.push_binary(OpKind::Mul, x, y)
    }
    fn div(&mut self, x: ExprId, y: ExprId) -> ExprId {
        self.push_binary(OpKind::Div, x, y)
    }
    fn sqrt(&mut self, x: ExprId) -> ExprId {
        self.push_unary(OpKind::Sqrt, x)
    }
    fn abs(&mut self, x: ExprId) -> ExprId {
        self.push_unary(OpKind::Abs, x)
    }
    fn floor(&mut self, x: ExprId) -> ExprId {
        self.push_unary(OpKind::Floor, x)
    }
    fn minv(&mut self, x: ExprId, y: ExprId) -> ExprId {
        self.push_binary(OpKind::Min, x, y)
    }
    fn maxv(&mut self, x: ExprId, y: ExprId) -> ExprId {
        self.push_binary(OpKind::Max, x, y)
    }
    fn sin(&mut self, x: ExprId) -> ExprId {
        self.push_unary(OpKind::Sin, x)
    }
    fn cos(&mut self, x: ExprId) -> ExprId {
        self.push_unary(OpKind::Cos, x)
    }
    fn atan2(&mut self, y: ExprId, x: ExprId) -> ExprId {
        self.push_binary(OpKind::Atan2, y, x)
    }
    fn ln(&mut self, x: ExprId) -> ExprId {
        self.push_unary(OpKind::Ln, x)
    }
    fn log2(&mut self, x: ExprId) -> ExprId {
        self.push_unary(OpKind::Log2, x)
    }
    fn gt(&mut self, x: ExprId, y: ExprId) -> ExprId {
        self.push_binary(OpKind::Gt, x, y)
    }
    fn ge(&mut self, x: ExprId, y: ExprId) -> ExprId {
        self.push_binary(OpKind::Ge, x, y)
    }
    fn mul_add(&mut self, x: ExprId, y: ExprId, z: ExprId) -> ExprId {
        self.push_ternary(OpKind::MulAdd, x, y, z)
    }
    fn select(&mut self, cond: ExprId, t: ExprId, f: ExprId) -> ExprId {
        self.push_ternary(OpKind::Select, cond, t, f)
    }
}

/// iq's cosine colour palette — `color(t) = a + b*cos(2*pi*(c*t+d))` — summed
/// over the three channels of iq's first published coefficient set
/// (a=b=0.5, c=1, d=0/0.33/0.67).
///
/// - Shader: "Cosine Color Palette" article / example
///   <https://www.shadertoy.com/view/Xl2GRc>
/// - Author: Inigo Quilez (iq)
/// - Source: <https://iquilezles.org/articles/palettes/>
/// - License: no explicit license stated on iquilezles.org; the linked
///   ShaderToy example's own declared license could not be confirmed
///   (ShaderToy's bot-verification blocks automated fetch), treated
///   conservatively as ShaderToy's site-wide default, CC BY-NC-SA 3.0.
/// - Fetched: 2026-08-27.
/// - Simplified: `t` is the radial distance from the origin (viewport-
///   clamped) rather than a 1D gradient parameter; the three RGB channels
///   are summed into one scalar rather than returned as a vec3 — this
///   corpus's kernels are single-channel.
fn cosine_palette() -> (ExprArena, ExprId) {
    fn channel(a: &mut ExprArena, t: ExprId, d: f32) -> ExprId {
        let dk = a.k(d);
        let td = a.add(t, dk);
        let two_pi = a.k(TAU);
        let arg = a.mul(td, two_pi);
        let c = a.cos(arg);
        let half = a.k(0.5);
        let hc = a.mul(c, half);
        a.add(hc, half)
    }
    let mut a = ExprArena::new();
    let x = a.viewport(0, 2.0);
    let y = a.viewport(1, 2.0);
    let t = a.length(x, y);
    let r = channel(&mut a, t, 0.0);
    let g = channel(&mut a, t, 0.33);
    let b = channel(&mut a, t, 0.67);
    let rg = a.add(r, g);
    let sum = a.add(rg, b);
    (a, sum)
}

/// iq's quadratic polynomial smooth-minimum, unioning two circle SDFs at
/// different centers — the composition iq's own article demonstrates.
///
/// - Shader: "smin (smooth minimum)" article / examples
///   <https://www.shadertoy.com/view/DlVcW1>
/// - Author: Inigo Quilez (iq)
/// - Source: <https://iquilezles.org/articles/smin/>
/// - License: no explicit license stated on iquilezles.org; the linked
///   ShaderToy example's declared license could not be confirmed (ShaderToy
///   blocks automated fetch), treated conservatively as CC BY-NC-SA 3.0.
/// - Fetched: 2026-08-27.
/// - Simplified: the two circle SDFs are this corpus's own primitive (same
///   form as the existing `circle_sdf` named kernel) rather than iq's own
///   demo shapes; only the `smin` formula itself is transcribed verbatim.
fn smooth_min_scene() -> (ExprArena, ExprId) {
    // `circle_at`: (center_x, center_y, radius), grouped into one tuple so
    // the builder stays under clippy's/CLAUDE.md's argument-count limit.
    fn circle(a: &mut ExprArena, x: ExprId, y: ExprId, circle_at: (f32, f32, f32)) -> ExprId {
        let (cx, cy, r) = circle_at;
        let cxk = a.k(cx);
        let cyk = a.k(cy);
        let dx = a.sub(x, cxk);
        let dy = a.sub(y, cyk);
        let d = a.length(dx, dy);
        let rk = a.k(r);
        a.sub(d, rk)
    }
    // iq: k *= 4.0; h = max(k-|a-b|,0)/k; return min(a,b) - h*h*k*(1/4).
    fn smin(a: &mut ExprArena, x: ExprId, y: ExprId, k: f32) -> ExprId {
        let kk = a.k(k * 4.0);
        let diff = a.sub(x, y);
        let ad = a.abs(diff);
        let kmad = a.sub(kk, ad);
        let zero = a.k(0.0);
        let hn = a.maxv(kmad, zero);
        let h = a.div(hn, kk);
        let h2 = a.sq(h);
        let h2k = a.mul(h2, kk);
        let quarter = a.k(0.25);
        let term = a.mul(h2k, quarter);
        let m = a.minv(x, y);
        a.sub(m, term)
    }
    let mut a = ExprArena::new();
    let x = a.viewport(0, 2.0);
    let y = a.viewport(1, 2.0);
    let c1 = circle(&mut a, x, y, (-0.35, 0.0, 0.55));
    let c2 = circle(&mut a, x, y, (0.35, 0.05, 0.5));
    let root = smin(&mut a, c1, c2, 0.3);
    (a, root)
}

/// Distance estimate to the Mandelbrot set via the Hubbard-Douady potential
/// `d = sqrt(m2/dz2) * 0.5 * ln(m2)`, fixed at 5 unrolled iterations. Escape
/// is a compare-and-freeze (`Select`) rather than a data-dependent loop
/// exit: pixelflow kernels have no branches, so a fixed iteration count
/// keeps updating past an escaped point unless something stops it — the
/// freeze makes that numerically harmless instead of a silently wrong
/// answer.
///
/// - Shader: "distance to the Mandelbrot set" article / example
///   <https://www.shadertoy.com/view/lsX3W4>
/// - Author: Inigo Quilez (iq)
/// - Source: <https://iquilezles.org/articles/distancefractals/>
/// - License: no explicit license stated on iquilezles.org; the linked
///   ShaderToy example's declared license could not be confirmed (ShaderToy
///   blocks automated fetch), treated conservatively as CC BY-NC-SA 3.0.
/// - Fetched: 2026-08-27.
/// - Simplified: 5 fixed unrolled iterations (the article's own examples
///   iterate until escape or a few hundred steps); escape is a `Select`
///   freeze rather than an early `break`; `c` is the pixel position clamped
///   into `[-2,2]^2` rather than driven by a pan/zoom camera transform.
fn mandelbrot_distance() -> (ExprArena, ExprId) {
    const ITERS: usize = 5;
    const BAILOUT2: f32 = 100.0;

    let mut a = ExprArena::new();
    let cx = a.viewport(0, 2.0);
    let cy = a.viewport(1, 2.0);

    let mut zx = a.k(0.0);
    let mut zy = a.k(0.0);
    let mut dzx = a.k(0.0);
    let mut dzy = a.k(0.0);
    let bail = a.k(BAILOUT2);
    let two = a.k(2.0);
    let one = a.k(1.0);

    for _ in 0..ITERS {
        let r2 = a.length2(zx, zy);
        let escaped = a.ge(r2, bail);

        // dz' = 2*(z*dz) + 1 (complex derivative recurrence).
        let zx_dzx = a.mul(zx, dzx);
        let zy_dzy = a.mul(zy, dzy);
        let re = a.sub(zx_dzx, zy_dzy);
        let re2 = a.mul(re, two);
        let new_dzx = a.add(re2, one);

        let zx_dzy = a.mul(zx, dzy);
        let zy_dzx = a.mul(zy, dzx);
        let im = a.add(zx_dzy, zy_dzx);
        let new_dzy = a.mul(im, two);

        // z = z*z + c.
        let zx2 = a.sq(zx);
        let zy2 = a.sq(zy);
        let re_z = a.sub(zx2, zy2);
        let new_zx = a.add(re_z, cx);
        let zxzy = a.mul(zx, zy);
        let im_z = a.mul(zxzy, two);
        let new_zy = a.add(im_z, cy);

        zx = a.select(escaped, zx, new_zx);
        zy = a.select(escaped, zy, new_zy);
        dzx = a.select(escaped, dzx, new_dzx);
        dzy = a.select(escaped, dzy, new_dzy);
    }

    let m2 = a.length2(zx, zy);
    let eps = a.k(1e-6);
    let m2s = a.maxv(m2, eps);
    let dz2 = a.length2(dzx, dzy);
    let dz2s = a.maxv(dz2, eps);
    let ratio = a.div(m2s, dz2s);
    let sr = a.sqrt(ratio);
    let lm = a.ln(m2s);
    let half = a.k(0.5);
    let hl = a.mul(lm, half);
    let root = a.mul(sr, hl);
    (a, root)
}

/// `sdPentagram` — a five-pointed star SDF, built from two reflections
/// across fixed axes plus a clamped-projection distance to the final edge.
///
/// - Shader: "2D distance functions" article — `sdPentagram` / example
///   <https://www.shadertoy.com/view/t3X3z4>
/// - Author: Inigo Quilez (iq)
/// - Source: <https://iquilezles.org/articles/distfunctions2d/>
/// - License: no explicit license stated on iquilezles.org; the linked
///   ShaderToy example's declared license could not be confirmed (ShaderToy
///   blocks automated fetch), treated conservatively as CC BY-NC-SA 3.0.
/// - Fetched: 2026-08-27.
/// - Simplified: `sign(...)` (not a pixelflow op) is replaced with a
///   `Ge`+`Select` producing +-1 — GLSL's `sign(0)==0` becomes `+1` here, a
///   difference only on the measure-zero set where the argument is exactly
///   zero.
fn star_sdf() -> (ExprArena, ExprId) {
    // p -= 2*max(dot(v,p),0)*v.
    fn fold(a: &mut ExprArena, px: ExprId, py: ExprId, vx: f32, vy: f32) -> (ExprId, ExprId) {
        let vxk = a.k(vx);
        let vyk = a.k(vy);
        let dpx = a.mul(px, vxk);
        let dpy = a.mul(py, vyk);
        let dot = a.add(dpx, dpy);
        let zero = a.k(0.0);
        let m = a.maxv(dot, zero);
        let two = a.k(2.0);
        let m2 = a.mul(m, two);
        let ox = a.mul(m2, vxk);
        let oy = a.mul(m2, vyk);
        (a.sub(px, ox), a.sub(py, oy))
    }

    const K1X: f32 = 0.809_017;
    const K2X: f32 = 0.309_017;
    const K1Y: f32 = 0.587_785_25;
    const K2Y: f32 = 0.951_056_5;
    const K1Z: f32 = 0.726_542_53;
    const R: f32 = 0.6;

    let mut a = ExprArena::new();
    let x0 = a.viewport(0, 2.0);
    let y0 = a.viewport(1, 2.0);
    let px0 = a.abs(x0);

    let (px1, py1) = fold(&mut a, px0, y0, K1X, -K1Y);
    let (px2, py2) = fold(&mut a, px1, py1, -K1X, -K1Y);
    let px3 = a.abs(px2);
    let rk = a.k(R);
    let py3 = a.sub(py2, rk);

    let v3xk = a.k(K2X);
    let v3yk = a.k(-K2Y);
    let dpx3 = a.mul(px3, v3xk);
    let dpy3 = a.mul(py3, v3yk);
    let dot3 = a.add(dpx3, dpy3);
    let clamped = a.clampf(dot3, 0.0, K1Z * R);
    let ox = a.mul(clamped, v3xk);
    let oy = a.mul(clamped, v3yk);
    let qx = a.sub(px3, ox);
    let qy = a.sub(py3, oy);
    let len = a.length(qx, qy);

    let t1 = a.mul(py3, v3xk);
    let t2 = a.mul(px3, v3yk);
    let cross = a.sub(t1, t2);
    let zero = a.k(0.0);
    let pos = a.ge(cross, zero);
    let one = a.k(1.0);
    let negone = a.k(-1.0);
    let sgn = a.select(pos, one, negone);
    let root = a.mul(len, sgn);
    (a, root)
}

/// Gyroid triply-periodic minimal surface, evaluated directly as a
/// 3-variable implicit field (`sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x)`)
/// with a second, higher-frequency detail octave — the closed-form
/// technique widely used in the raymarching/SDF shader community as an
/// infill/lattice primitive.
///
/// - Shader: "gyroid SDF"
///   <https://www.shadertoy.com/view/wtfSRS>
/// - Author: zzggbb (ShaderToy handle), posted 2019-07-17.
/// - License: ShaderToy default (CC BY-NC-SA 3.0); the shader's own license
///   declaration could not be confirmed — ShaderToy's bot-verification
///   blocks automated fetch of the page.
/// - Fetched: 2026-08-27.
/// - Simplified: the raymarch loop is dropped entirely — this evaluates the
///   implicit gyroid field directly at (x, y, z), a true closed-form 2D
///   slice rather than a stepped ray. Exact GLSL was not transcribed (fetch
///   blocked); this reimplements the standard gyroid formula the cited
///   shader is an example of. A second, higher-frequency/lower-amplitude
///   gyroid term is layered in as detail (a common technique in gyroid
///   infill shaders). The raw implicit value is returned rather than the
///   corrected pseudo-distance some implementations multiply in.
fn gyroid_slice() -> (ExprArena, ExprId) {
    fn gyroid(a: &mut ExprArena, x: ExprId, y: ExprId, z: ExprId) -> ExprId {
        let sx = a.sin(x);
        let cy = a.cos(y);
        let sy = a.sin(y);
        let cz = a.cos(z);
        let sz = a.sin(z);
        let cx = a.cos(x);
        let t1 = a.mul(sx, cy);
        let t2 = a.mul(sy, cz);
        let t3 = a.mul(sz, cx);
        let s = a.add(t1, t2);
        a.add(s, t3)
    }
    let mut a = ExprArena::new();
    let x = a.viewport(0, 6.0);
    let y = a.viewport(1, 6.0);
    let z = a.arg(0.0);

    let g1 = gyroid(&mut a, x, y, z);

    let detail_f = a.k(3.0);
    let xd = a.mul(x, detail_f);
    let yd = a.mul(y, detail_f);
    let zd = a.mul(z, detail_f);
    let g2 = gyroid(&mut a, xd, yd, zd);

    let amp = a.k(0.25);
    let g2s = a.mul(g2, amp);
    let root = a.add(g1, g2s);
    (a, root)
}

/// Classic multi-sine "plasma" effect: axis-aligned, diagonal, and radial
/// sine waves summed and animated by a time parameter.
///
/// - Shader: "Plasma 90x" ("an oldskool plasma effect")
///   <https://www.shadertoy.com/view/4ssGR7>
/// - Author: bitek (ShaderToy handle), posted 2013-04-16.
/// - License: ShaderToy default (CC BY-NC-SA 3.0); the shader's own license
///   declaration could not be confirmed — ShaderToy's bot-verification
///   blocks automated fetch of the page.
/// - Fetched: 2026-08-27.
/// - Simplified: exact GLSL was not transcribed (fetch blocked); this
///   reimplements the standard four-term plasma sum (axis + diagonal +
///   radial sines) the cited shader is a classic example of, rather than
///   its specific palette/post-processing. The kernel's clock — a uniform,
///   because it is one value for a whole frame — stands in for ShaderToy's
///   `iTime`.
fn plasma() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.viewport(0, 6.0);
    let y = a.viewport(1, 6.0);
    let t = a.arg(0.0);

    let f1 = a.k(1.0);
    let xt = a.mul(x, f1);
    let term1arg = a.add(xt, t);
    let term1 = a.sin(term1arg);

    let f2 = a.k(1.3);
    let yt = a.mul(y, f2);
    let term2arg = a.add(yt, t);
    let term2 = a.sin(term2arg);

    let f3 = a.k(0.7);
    let xy = a.add(x, y);
    let xyt = a.mul(xy, f3);
    let term3arg = a.add(xyt, t);
    let term3 = a.sin(term3arg);

    let r = a.length(x, y);
    let f4 = a.k(1.7);
    let rt = a.mul(r, f4);
    let term4arg = a.add(rt, t);
    let term4 = a.sin(term4arg);

    let s12 = a.add(term1, term2);
    let s123 = a.add(s12, term3);
    let s1234 = a.add(s123, term4);
    let quarter = a.k(0.25);
    let root = a.mul(s1234, quarter);
    (a, root)
}

/// Domain-warped fractional-Brownian-motion pattern:
/// `f(p) = fbm(p + k*fbm(p + offset))`.
///
/// - Shader: "Domain Warping" article / example
///   <https://www.shadertoy.com/view/4s23zzM>
/// - Author: Inigo Quilez (iq)
/// - Source: <https://iquilezles.org/articles/warp/>
/// - License: no explicit license stated on iquilezles.org; the linked
///   ShaderToy example's declared license could not be confirmed (ShaderToy
///   blocks automated fetch), treated conservatively as CC BY-NC-SA 3.0.
/// - Fetched: 2026-08-27.
/// - Simplified: the article's value/gradient noise (texture- or
///   hash-function-backed) is replaced by the standard
///   `fract(sin(dot(p,k))*c)` GLSL pseudo-hash idiom (no textures/buffers
///   are representable in this e-graph); fbm is truncated to 2 octaves; the
///   warp is a single level with a scalar offset field reused for both
///   coordinates. The article's own recipe warps twice with a
///   2-component field each time — this port keeps only the first level,
///   to stay inside the corpus's ~30-400 node band (see the module doc's
///   "what was tried and abandoned").
/// - Numerically ill-conditioned by construction, documented rather than
///   "fixed": `hash()`'s `fract(sin(d)*43758.547)` multiplies `sin(d)` by a
///   huge constant *before* subtracting `floor`, so any conforming
///   evaluator's rounding latitude that CLAUDE.md's "Floating point at the
///   edges" explicitly licenses (`MulAdd` fusing one rounding instead of
///   two; an e-graph-extracted form associating `Add(Mul,Mul)` differently)
///   is amplified past `floor`/`fract`'s cancellation into an output that
///   two independently-correct evaluators are not guaranteed to agree on —
///   occasionally not even to the same integer. This is the *only* kernel
///   in this corpus using that idiom, which is why `egraph_off_on`'s raw
///   `oracle()` diff reports it an order of magnitude above every other
///   kernel; it is not a compiler defect (see
///   `pixelflow-pipeline/tests/domain_warp_fbm_hash_conditioning.rs`, whose
///   module doc has the full bisection against an independent `f64`
///   reference, and which is the check to consult — or extend — before
///   treating a large `same_form`/`cross_form` number here as a miscompile).
fn domain_warp_fbm() -> (ExprArena, ExprId) {
    fn hash(a: &mut ExprArena, x: ExprId, y: ExprId) -> ExprId {
        let kx = a.k(127.1);
        let ky = a.k(311.7);
        let xkx = a.mul(x, kx);
        let yky = a.mul(y, ky);
        let d = a.add(xkx, yky);
        let s = a.sin(d);
        let c = a.k(43_758.547);
        let v = a.mul(s, c);
        let f = a.floor(v);
        a.sub(v, f) // fract(v) in [0, 1).
    }
    fn fbm2(a: &mut ExprArena, x: ExprId, y: ExprId) -> ExprId {
        let n0 = hash(a, x, y);
        let two = a.k(2.0);
        let x2 = a.mul(x, two);
        let y2 = a.mul(y, two);
        let ox = a.k(7.3);
        let oy = a.k(1.7);
        let x2o = a.add(x2, ox);
        let y2o = a.add(y2, oy);
        let n1 = hash(a, x2o, y2o);
        let half = a.k(0.5);
        let quarter = a.k(0.25);
        let n0h = a.mul(n0, half);
        let n1q = a.mul(n1, quarter);
        a.add(n0h, n1q)
    }
    let mut a = ExprArena::new();
    let x = a.viewport(0, 4.0);
    let y = a.viewport(1, 4.0);

    let ox = a.k(5.2);
    let oy = a.k(1.3);
    let xo = a.add(x, ox);
    let yo = a.add(y, oy);
    let q = fbm2(&mut a, xo, yo);

    let kwarp = a.k(4.0);
    let qk = a.mul(q, kwarp);
    let wx = a.add(x, qk);
    let oy2 = a.k(3.1);
    let qk_oy = a.add(qk, oy2);
    let wy = a.add(y, qk_oy);

    let root = fbm2(&mut a, wx, wy);
    (a, root)
}

/// N-fold kaleidoscope: fold the polar angle into a repeating wedge (mirror
/// symmetry via `floor` + `abs`), reconstruct Cartesian coordinates, and
/// evaluate a striped pattern with a hard-threshold `Select` branch plus a
/// radial ring modulation.
///
/// - Shader: "Kaleidoscope Tutorial"
///   <https://www.shadertoy.com/view/WdcSRr>
/// - Author: deliaev (ShaderToy handle), posted 2020-07-22.
/// - License: ShaderToy default (CC BY-NC-SA 3.0); the shader's own license
///   declaration could not be confirmed — ShaderToy's bot-verification
///   blocks automated fetch of the page.
/// - Fetched: 2026-08-27.
/// - Simplified: exact GLSL was not transcribed (fetch blocked); this
///   reimplements the standard polar-fold kaleidoscope technique (angle
///   mod-folded into a wedge, then mirrored) the cited shader teaches, with
///   its own simple striped/thresholded pattern in place of a sampled
///   texture.
fn kaleidoscope_fold() -> (ExprArena, ExprId) {
    const SEGMENTS: f32 = 6.0;

    let mut a = ExprArena::new();
    let x = a.viewport(0, 4.0);
    let y = a.viewport(1, 4.0);

    let r = a.length(x, y);
    let theta = a.atan2(y, x);

    // Fold theta into [-seg/2, seg/2]: theta - seg*floor(theta/seg + 0.5).
    let seg = a.k(TAU / SEGMENTS);
    let half = a.k(0.5);
    let ts = a.div(theta, seg);
    let tsh = a.add(ts, half);
    let n = a.floor(tsh);
    let ns = a.mul(n, seg);
    let folded = a.sub(theta, ns);
    let mirrored = a.abs(folded);

    let cm = a.cos(mirrored);
    let sm = a.sin(mirrored);
    let fx = a.mul(r, cm);
    let fy = a.mul(r, sm);

    let freq = a.k(8.0);
    let fxf = a.mul(fx, freq);
    let fyf = a.mul(fy, freq);
    let sp = a.sin(fxf);
    let cp = a.cos(fyf);
    let pattern = a.mul(sp, cp);

    let zero = a.k(0.0);
    let branch_cond = a.gt(pattern, zero);
    let one = a.k(1.0);
    let negone = a.k(-1.0);
    let branch = a.select(branch_cond, one, negone);
    let stripe = a.mul(branch, pattern); // == |pattern|, via a genuine Select.

    let ring_f = a.k(5.0);
    let rf = a.mul(r, ring_f);
    let ring = a.sin(rf);

    let blend = a.k(0.5);
    let stripe_h = a.mul(stripe, blend);
    let ring_h = a.mul(ring, blend);
    let root = a.add(stripe_h, ring_h);
    (a, root)
}

/// Three-ball metaball field: sum of inverse-square "energy" contributions,
/// thresholded to a hard silhouette (`Select`) and blended with a softened
/// version of the same threshold.
///
/// - Shader: "Metaball", posted 2013 (a simple 2D metaballs shader).
///   <https://www.shadertoy.com/view/Xdl3Wl>
/// - Author: not resolved — ShaderToy's bot-verification blocks automated
///   fetch of the page, and the handle did not surface via web search.
/// - License: ShaderToy default (CC BY-NC-SA 3.0), unconfirmed per-shader.
/// - Fetched: 2026-08-27.
/// - Technique origin: the underlying metaball/"blobby object" field is Jim
///   Blinn's ("A Generalization of Algebraic Surface Drawing," ACM ToG
///   1(3), 1982) — this port implements that classic sum-of-fields
///   technique (which the cited shader is one of many modern examples of)
///   rather than transcribing the cited shader's exact GLSL (not
///   fetchable).
fn metaballs() -> (ExprArena, ExprId) {
    // `ball_at`: (center_x, center_y, radius_squared), grouped into one
    // tuple so the builder stays under clippy's/CLAUDE.md's argument-count
    // limit.
    fn ball(a: &mut ExprArena, x: ExprId, y: ExprId, ball_at: (f32, f32, f32)) -> ExprId {
        let (cx, cy, r2) = ball_at;
        let cxk = a.k(cx);
        let cyk = a.k(cy);
        let dx = a.sub(x, cxk);
        let dy = a.sub(y, cyk);
        let d2 = a.length2(dx, dy);
        let eps = a.k(1e-3);
        let d2s = a.maxv(d2, eps);
        let r2k = a.k(r2);
        a.div(r2k, d2s)
    }
    let mut a = ExprArena::new();
    let x = a.viewport(0, 3.0);
    let y = a.viewport(1, 3.0);

    let e1 = ball(&mut a, x, y, (-0.6, 0.0, 0.25));
    let e2 = ball(&mut a, x, y, (0.5, 0.3, 0.2));
    let e3 = ball(&mut a, x, y, (0.0, -0.5, 0.18));
    let e12 = a.add(e1, e2);
    let field = a.add(e12, e3);

    let threshold = a.k(1.0);
    let hard_cond = a.ge(field, threshold);
    let one = a.k(1.0);
    let zero = a.k(0.0);
    let hard = a.select(hard_cond, one, zero);

    // Poor-man's smoothstep-ish soft blend: clamp((field-thr)*4+0.5, 0, 1).
    let diff = a.sub(field, threshold);
    let four = a.k(4.0);
    let scaled = a.mul(diff, four);
    let halfk = a.k(0.5);
    let biased = a.add(scaled, halfk);
    let soft = a.clampf(biased, 0.0, 1.0);

    let hardh = a.mul(hard, halfk);
    let softh = a.mul(soft, halfk);
    let root = a.add(hardh, softh);
    (a, root)
}

/// Cubic Julia set (`z -> z^3 + c` for FIXED c, `z0` = pixel position), fixed
/// at 5 unrolled iterations with escape-freeze, smooth-colored from the
/// final (frozen) modulus via the classic `n - log2(log2(|z|))`-family
/// escape-time estimate.
///
/// - Shader: "Julia - Distance 2" — "an SDF for the Julia set of f(z)=z^3+C"
///   <https://www.shadertoy.com/view/3llyzl>
/// - Author: Inigo Quilez (iq)
/// - License: CC BY-NC-SA 3.0 (confirmed).
/// - Fetched: 2026-08-27.
/// - Simplified: 5 fixed unrolled iterations rather than a data-dependent
///   distance-estimate raymarch; escape is a `Select` freeze, not an early
///   `break`; smooth coloring is evaluated once from the final frozen z
///   rather than iq's true continuous per-iteration distance estimate (this
///   kernel has no loop-carried iteration counter to make that exact).
///   Exact GLSL was not transcribed (ShaderToy fetch blocked) — this
///   reimplements the standard cubic-Julia escape-time algorithm.
fn julia_set() -> (ExprArena, ExprId) {
    const ITERS: usize = 5;
    const BAILOUT2: f32 = 100.0;
    const CX: f32 = -0.4;
    const CY: f32 = 0.6;

    let mut a = ExprArena::new();
    let mut zx = a.viewport(0, 1.6);
    let mut zy = a.viewport(1, 1.6);
    let cx = a.k(CX);
    let cy = a.k(CY);
    let bail = a.k(BAILOUT2);
    let two = a.k(2.0);

    for _ in 0..ITERS {
        let r2 = a.length2(zx, zy);
        let escaped = a.ge(r2, bail);

        // z^2 = (zx^2 - zy^2, 2*zx*zy).
        let zx2 = a.sq(zx);
        let zy2 = a.sq(zy);
        let z2x = a.sub(zx2, zy2);
        let zxzy = a.mul(zx, zy);
        let z2y = a.mul(zxzy, two);

        // z^3 = z^2 * z.
        let a1 = a.mul(z2x, zx);
        let a2 = a.mul(z2y, zy);
        let z3x = a.sub(a1, a2);
        let a3 = a.mul(z2x, zy);
        let a4 = a.mul(z2y, zx);
        let z3y = a.add(a3, a4);

        let new_zx = a.add(z3x, cx);
        let new_zy = a.add(z3y, cy);

        zx = a.select(escaped, zx, new_zx);
        zy = a.select(escaped, zy, new_zy);
    }

    let r2f = a.length2(zx, zy);
    let eps = a.k(1.0001);
    let r2s = a.maxv(r2f, eps);
    let l1 = a.log2(r2s);
    let halfk = a.k(0.5);
    let l1h = a.mul(l1, halfk);
    let floor_eps = a.k(0.0001);
    let l1he = a.maxv(l1h, floor_eps);
    let l2 = a.log2(l1he);
    let itersk = a.k(ITERS as f32);
    let root = a.sub(itersk, l2);
    (a, root)
}

/// Radial vignette plus a thin highlight ring, both built from the standard
/// GLSL `smoothstep` — the cubic Hermite `t*t*(3-2t)` curve — evaluated with
/// `MulAdd` so `(3-2t)` is a genuine fused multiply-add.
///
/// - Source: "smoothstep" glossary entry
///   <https://thebookofshaders.com/glossary/?search=smoothstep>
/// - Author: Patricio Gonzalez Vivo (The Book of Shaders), copyright 2015.
/// - License: Book of Shaders site-wide license (Creative Commons
///   Attribution-NonCommercial-ShareAlike); the exact version was not
///   visible in the fetched excerpt.
/// - Fetched: 2026-08-27.
/// - Simplified: this is the bare `smoothstep` primitive applied directly
///   as a radial falloff and a ring, not transcribed from any single
///   specific ShaderToy vignette shader — `smoothstep`-as-vignette is the
///   standard, widely-taught technique the glossary entry itself describes.
fn smoothstep_vignette() -> (ExprArena, ExprId) {
    // smoothstep(e0, e1, x) = let t = clamp((x-e0)/(e1-e0), 0, 1) in
    // t*t*(3-2t), with (3-2t) computed as MulAdd(t, -2, 3).
    fn smoothstep(a: &mut ExprArena, e0: f32, e1: f32, x: ExprId) -> ExprId {
        let e0k = a.k(e0);
        let e1k = a.k(e1);
        let num = a.sub(x, e0k);
        let den = a.sub(e1k, e0k);
        let raw = a.div(num, den);
        let t = a.clampf(raw, 0.0, 1.0);
        let negtwo = a.k(-2.0);
        let three = a.k(3.0);
        let poly = a.mul_add(t, negtwo, three); // 3 - 2t
        let t2 = a.sq(t);
        a.mul(t2, poly)
    }

    let mut a = ExprArena::new();
    let x = a.viewport(0, 3.0);
    let y = a.viewport(1, 3.0);
    let r = a.length(x, y);

    let vig_ss = smoothstep(&mut a, 0.6, 1.8, r);
    let one = a.k(1.0);
    let vig = a.sub(one, vig_ss);

    let ring_in = smoothstep(&mut a, 0.9, 1.0, r);
    let ring_out = smoothstep(&mut a, 1.0, 1.1, r);
    let ring = a.sub(ring_in, ring_out);

    let vig_w = a.k(0.7);
    let ring_w = a.k(0.3);
    let vigw = a.mul(vig, vig_w);
    let ringw = a.mul(ring, ring_w);
    let root = a.add(vigw, ringw);
    (a, root)
}

/// Torus SDF (`sdTorus`), evaluated as a genuine 3-input field over (X, Y,
/// Z) — a true slice through the 3D shape needs no raymarch, the distance
/// function is closed-form. Unioned (`min`) with a second, smaller torus for
/// scene composition.
///
/// - Shader: "Signed Distance Functions" article — `sdTorus`, playlist
///   <https://www.shadertoy.com/playlist/43cXRl>
/// - Author: Inigo Quilez (iq)
/// - Source: <https://iquilezles.org/articles/distfunctions/>
/// - License: no explicit license stated on iquilezles.org; the linked
///   ShaderToy playlist's per-shader licenses could not be confirmed
///   (ShaderToy blocks automated fetch), treated conservatively as
///   CC BY-NC-SA 3.0.
/// - Fetched: 2026-08-27.
/// - Simplified: no camera/raymarch — the SDF is evaluated directly at
///   (x, y, z) as a genuine 3-variable field, in place of stepping a ray
///   through it; the second torus (union) is this port's own addition for
///   scene composition, not part of the cited article.
fn torus_slice() -> (ExprArena, ExprId) {
    // `p`: (x, y, z); `radii`: (major, minor) — grouped into tuples so the
    // builder stays under clippy's/CLAUDE.md's argument-count limit.
    fn torus(a: &mut ExprArena, p: (ExprId, ExprId, ExprId), radii: (f32, f32)) -> ExprId {
        let (x, y, z) = p;
        let (major, minor) = radii;
        let qx = a.length(x, z);
        let majork = a.k(major);
        let qx2 = a.sub(qx, majork);
        let d = a.length(qx2, y);
        let minork = a.k(minor);
        a.sub(d, minork)
    }
    let mut a = ExprArena::new();
    let x = a.viewport(0, 3.0);
    let y = a.viewport(1, 3.0);
    let z = a.arg(0.0);

    let t1 = torus(&mut a, (x, y, z), (1.0, 0.35));

    let zoff = a.k(0.8);
    let z2 = a.sub(z, zoff);
    let t2 = torus(&mut a, (x, y, z2), (0.5, 0.15));

    let root = a.minv(t1, t2);
    (a, root)
}

/// Names this module's kernels are registered under in
/// `corpus_split.toml`'s `[final].kernels` and `gen_bench_corpus`'s
/// `named_kernel()` — the same touch points the five original named kernels
/// use.
pub const SHADERTOY_KERNEL_NAMES: [&str; 12] = [
    "cosine_palette",
    "smooth_min_scene",
    "mandelbrot_distance",
    "star_sdf",
    "gyroid_slice",
    "plasma",
    "domain_warp_fbm",
    "kaleidoscope_fold",
    "metaballs",
    "julia_set",
    "smoothstep_vignette",
    "torus_slice",
];

// ============================================================================
// The five original named production kernels (FINAL tier). They predate the
// ShaderToy set — `pixelflow-search/examples/rule_report.rs` carries the
// same five — and were duplicated into `gen_bench_corpus` until the
// structural-gap inventory (`corpus_gaps`) needed them too; one definition
// here now serves every harness.
// ============================================================================

/// The five original named production kernels, in manifest order.
pub const NAMED_KERNEL_NAMES: [&str; 5] = ["swirl", "circle_sdf", "poly", "redundant", "normalize"];

/// sin(sqrt(x*x + y*y) * freq) * amp + bias — the swirl shader core.
fn swirl() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let xx = a.push_binary(OpKind::Mul, x, x);
    let yy = a.push_binary(OpKind::Mul, y, y);
    let d = a.push_binary(OpKind::Add, xx, yy);
    let s = a.push_unary(OpKind::Sqrt, d);
    let kf = a.push_const(3.0);
    let sf = a.push_binary(OpKind::Mul, s, kf);
    let sn = a.push_unary(OpKind::Sin, sf);
    let ka = a.push_const(0.5);
    let prod = a.push_binary(OpKind::Mul, sn, ka);
    let kb = a.push_const(0.5);
    let out = a.push_binary(OpKind::Add, prod, kb);
    (a, out)
}

/// Circle SDF: sqrt((x-cx)^2 + (y-cy)^2) - r.
fn circle_sdf() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let cx = a.push_const(0.3);
    let cy = a.push_const(-0.2);
    let dx = a.push_binary(OpKind::Sub, x, cx);
    let dy = a.push_binary(OpKind::Sub, y, cy);
    let dx2 = a.push_binary(OpKind::Mul, dx, dx);
    let dy2 = a.push_binary(OpKind::Mul, dy, dy);
    let sum = a.push_binary(OpKind::Add, dx2, dy2);
    let dist = a.push_unary(OpKind::Sqrt, sum);
    let r = a.push_const(0.5);
    let out = a.push_binary(OpKind::Sub, dist, r);
    (a, out)
}

/// FMA-bait polynomial: a*x*x + b*x + c (Horner-able, fusion-able).
fn poly() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let ka = a.push_const(2.0);
    let kb = a.push_const(-3.0);
    let kc = a.push_const(1.0);
    let xx = a.push_binary(OpKind::Mul, x, x);
    let ax2 = a.push_binary(OpKind::Mul, ka, xx);
    let bx = a.push_binary(OpKind::Mul, kb, x);
    let s1 = a.push_binary(OpKind::Add, ax2, bx);
    let out = a.push_binary(OpKind::Add, s1, kc);
    (a, out)
}

/// Redundancy bait: (x+y)*(x+y) + 2*(x+y) — CSE + distribution territory.
fn redundant() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let s = a.push_binary(OpKind::Add, x, y);
    let s2 = a.push_binary(OpKind::Mul, s, s);
    let two = a.push_const(2.0);
    let ts = a.push_binary(OpKind::Mul, two, s);
    let out = a.push_binary(OpKind::Add, s2, ts);
    (a, out)
}

/// Division/sqrt bait: x / sqrt(x*x + y*y) (normalize — rsqrt rewrites).
fn normalize() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let xx = a.push_binary(OpKind::Mul, x, x);
    let yy = a.push_binary(OpKind::Mul, y, y);
    let d = a.push_binary(OpKind::Add, xx, yy);
    let s = a.push_unary(OpKind::Sqrt, d);
    let out = a.push_binary(OpKind::Div, x, s);
    (a, out)
}

/// Resolve any manifest kernel name — one of [`NAMED_KERNEL_NAMES`] or of
/// [`SHADERTOY_KERNEL_NAMES`] — to its arena builder.
#[must_use]
pub fn named_kernel(name: &str) -> Option<(ExprArena, ExprId)> {
    match name {
        "swirl" => Some(swirl()),
        "circle_sdf" => Some(circle_sdf()),
        "poly" => Some(poly()),
        "redundant" => Some(redundant()),
        "normalize" => Some(normalize()),
        _ => named_shadertoy_kernel(name),
    }
}

/// Resolve a name from [`SHADERTOY_KERNEL_NAMES`] to its arena builder.
/// [`named_kernel`] falls back to this for any name outside the five
/// original production kernels.
#[must_use]
pub fn named_shadertoy_kernel(name: &str) -> Option<(ExprArena, ExprId)> {
    match name {
        "cosine_palette" => Some(cosine_palette()),
        "smooth_min_scene" => Some(smooth_min_scene()),
        "mandelbrot_distance" => Some(mandelbrot_distance()),
        "star_sdf" => Some(star_sdf()),
        "gyroid_slice" => Some(gyroid_slice()),
        "plasma" => Some(plasma()),
        "domain_warp_fbm" => Some(domain_warp_fbm()),
        "kaleidoscope_fold" => Some(kaleidoscope_fold()),
        "metaballs" => Some(metaballs()),
        "julia_set" => Some(julia_set()),
        "smoothstep_vignette" => Some(smoothstep_vignette()),
        "torus_slice" => Some(torus_slice()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_declared_name_resolves() {
        for name in SHADERTOY_KERNEL_NAMES {
            assert!(
                named_shadertoy_kernel(name).is_some(),
                "{name} must resolve to a builder"
            );
        }
        assert!(named_shadertoy_kernel("does_not_exist").is_none());
    }

    #[test]
    fn every_named_production_kernel_resolves() {
        for name in NAMED_KERNEL_NAMES {
            assert!(named_kernel(name).is_some(), "{name} must resolve");
            assert!(
                named_shadertoy_kernel(name).is_none(),
                "{name} is not a ShaderToy port"
            );
        }
        assert!(named_kernel("julia_set").is_some());
        assert!(named_kernel("does_not_exist").is_none());
    }

    #[test]
    fn names_are_unique() {
        for (i, a) in SHADERTOY_KERNEL_NAMES.iter().enumerate() {
            for b in &SHADERTOY_KERNEL_NAMES[i + 1..] {
                assert_ne!(a, b, "duplicate kernel name {a}");
            }
        }
    }

    #[test]
    fn node_counts_stay_in_the_corpus_realistic_band() {
        // `arena.len()` — the arena's actual entry count — not
        // `node_count_subtree` (which re-walks a shared node once per
        // *reference* and is meant for BwdGenerator's largely tree-shaped
        // synthetic output). The fractal kernels here deliberately reuse
        // per-iteration state across several downstream consumers — real
        // sharing that both the JIT (`compile`) and the scalar
        // oracle's memoized evaluator (`eval.rs`'s per-`ExprId` memo table)
        // compile/evaluate once each, so `arena.len()` is what "expression
        // size" actually means for these — `node_count_subtree` explodes
        // combinatorially (a fully-unrolled-tree count) on exactly this
        // pattern without describing anything real about compiled cost.
        for name in SHADERTOY_KERNEL_NAMES {
            let (arena, _root) = named_shadertoy_kernel(name).expect("known kernel");
            let n = arena.len();
            assert!(
                (15..=400).contains(&n),
                "{name}: {n} arena nodes outside the corpus's realistic band \
                 (~30-400, generous floor of 15 for the simplest ports)"
            );
        }
    }
}
