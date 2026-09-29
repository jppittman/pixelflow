//! A quadratic arc that turns back in neither `x` nor `y`, and the host
//! split that is the only way to make one.
//!
//! ## What it is for
//!
//! The area left of a curve, integrated over a pixel, closes in one formula
//! only when the curve is a graph over `y` that also rises one way in `x`:
//! then `clamp(x(y) − X, 0, 1)` is zero, then linear, then one, along the
//! arc (docs/plans/2026-09-23-a-glyph-is-a-formula.md; the step-5 design's
//! `ArcMoment`). That is a fact about the table's numbers, which a rewrite
//! rule cannot see, so the kernel writes it in as a certificate —
//! `max(step, 0)` on each control-polygon step — and the formula becomes an
//! identity for every value in the table. The certificate is only *the
//! glyph's* geometry if every arc in the table already satisfies it. This
//! type is how the table's builder proves that: a [`MonotoneQuad`] can only
//! be built by [`MonotoneQuad::split`], which cuts any quadratic at its
//! interior extrema.
//!
//! ## Who cuts
//!
//! `loop_blinn`'s host turns every segment into pieces through this split
//! — a line too, as the quadratic whose control point is its midpoint,
//! which nothing cuts. The crate's own font has 995 quadratics with an
//! interior extremum (1022 turning points), in 252 simple glyphs (`ƕ`, `ȡ`,
//! `ʓ`, …), none in ASCII or Latin-1. A quadratic whose ends coincide is
//! dropped *before* the split: it has a cusp at `t = ½` on both axes, and
//! its two halves would retrace each other, cancelling only to rounding.

/// A point, `[x, y]`.
type P = [f64; 2];

/// How close to an end of the arc a cut may fall and still be taken: 2⁻³².
///
/// A cut closer than this would leave a piece shorter than 2⁻³² of the arc
/// — 2.3e-7 px for a 1000-px arc, far below the `f32` resolution the rows
/// are stored at. Not taking it leaves the arc turning back by at most that
/// fraction of its control polygon, and [`certified`] removes the rest. The
/// same margin decides when an `x`- and a `y`-extremum are one cusp: two
/// turning points closer than a cut can resolve are one turning point.
const SPLIT_MARGIN: f64 = 1.0 / 4_294_967_296.0;

/// A quadratic Bézier `p0 → p1 → p2` that is monotone in both coordinates.
/// On each axis its two control-polygon steps, `p1 − p0` and `p2 − p1`, never
/// have opposite signs, so the arc never turns back in `x` or in `y`.
///
/// For a quadratic, "the steps agree in sign" is exactly "no interior
/// extremum". The derivative `2((1 − t)(p1 − p0) + t(p2 − p1))` changes sign
/// on `(0, 1)` precisely when the steps do. That is also exactly the kernel's
/// certificate, so this is the geometry the certificate describes and not a
/// stronger or weaker property.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MonotoneQuad([P; 3]);

impl MonotoneQuad {
    /// `p0 → p1 → p2` cut at its interior extrema: one, two or three arcs, in
    /// order, the first starting at `p0`, the last ending at `p2`, each
    /// ending exactly where the next begins.
    ///
    /// **The cut.** An axis has an interior extremum exactly when its steps
    /// have opposite signs. That is tested on the signs themselves, so no
    /// product is rounded into zero. The extremum is at
    /// `t = s₀ / (s₀ − s₁)`. De Casteljau cuts there. On the axis that turns,
    /// both new control points are then set to the cut point's coordinate.
    /// That is where exact arithmetic would put them, since the tangent is
    /// flat on that axis at an extremum. Each half then has one zero step
    /// and one step that points the arc's way.
    ///
    /// **Two cuts at most, planned once.** Both parameters come from the
    /// quadratic as given, and neither is re-derived from a piece. A split
    /// that re-tests its own output can loop forever. At a near-cusp the two
    /// parameters differ by an ulp; cutting at one leaves a residual step of
    /// about 1e-16 on the other axis, whose extremum then rounds to `t = 1`.
    /// That found step 5's host split looping on
    /// `(3.126…, −3.695…) (−0.462…, 0.276…) (0.257…, −0.521…)`. Two turning
    /// points closer than [`SPLIT_MARGIN`] are one cusp, cut once with both
    /// axes flattened. A cut within [`SPLIT_MARGIN`] of the end of the arc it
    /// would cut is not taken.
    ///
    /// **Then the certificate.** Whatever sign a rounding left on a step is
    /// clamped away by [`certified`]. So every piece is monotone by
    /// construction, not by a test that could fail.
    #[must_use]
    pub(crate) fn split(p0: P, p1: P, p2: P) -> Vec<Self> {
        let whole = [p0, p1, p2];
        let mut pieces = Vec::with_capacity(3);
        let mut rest = whole;
        // Where `rest` starts, as a parameter of `whole`.
        let mut done = 0.0;
        for cut in cuts(whole) {
            let local = (cut.t - done) / (1.0 - done);
            if !(SPLIT_MARGIN..=1.0 - SPLIT_MARGIN).contains(&local) {
                continue;
            }
            let (left, right) = cut_at(rest, local, cut.flattens);
            pieces.push(left);
            rest = right;
            done = cut.t;
        }
        pieces.push(rest);
        pieces.into_iter().map(|q| Self(certified(q))).collect()
    }

    /// The control points, `[p0, p1, p2]`.
    #[must_use]
    pub(crate) fn points(self) -> [P; 3] {
        self.0
    }
}

/// Which coordinates turn at a cut, and so are flattened there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Turning {
    X,
    Y,
    /// Both at one parameter: a cusp, where the arc stops and reverses.
    Both,
}

impl Turning {
    fn axes(self) -> &'static [usize] {
        match self {
            Self::X => &[0],
            Self::Y => &[1],
            Self::Both => &[0, 1],
        }
    }
}

/// A planned cut, at a parameter of the whole arc.
#[derive(Clone, Copy, Debug)]
struct Cut {
    t: f64,
    flattens: Turning,
}

/// Where `q` turns back on `axis`, if it does inside `(0, 1)`.
fn turning_point(q: [P; 3], axis: usize) -> Option<f64> {
    let (s0, s1) = (q[1][axis] - q[0][axis], q[2][axis] - q[1][axis]);
    let opposite = (s0 < 0.0 && s1 > 0.0) || (s0 > 0.0 && s1 < 0.0);
    opposite.then(|| s0 / (s0 - s1))
}

/// Every cut `q` needs, in order: none, one, or two.
fn cuts(q: [P; 3]) -> Vec<Cut> {
    let at = |t, flattens| Cut { t, flattens };
    match (turning_point(q, 0), turning_point(q, 1)) {
        (None, None) => vec![],
        (Some(t), None) => vec![at(t, Turning::X)],
        (None, Some(t)) => vec![at(t, Turning::Y)],
        (Some(tx), Some(ty)) if (tx - ty).abs() <= SPLIT_MARGIN => {
            vec![at(0.5 * (tx + ty), Turning::Both)]
        }
        (Some(tx), Some(ty)) if tx < ty => vec![at(tx, Turning::X), at(ty, Turning::Y)],
        (Some(tx), Some(ty)) => vec![at(ty, Turning::Y), at(tx, Turning::X)],
    }
}

/// De Casteljau at `t`, with the coordinates that turn there flattened
/// onto the cut point.
fn cut_at(q: [P; 3], t: f64, flattens: Turning) -> ([P; 3], [P; 3]) {
    let lerp = |a: P, b: P| [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
    let (mut l, mut r) = (lerp(q[0], q[1]), lerp(q[1], q[2]));
    let m = lerp(l, r);
    for &axis in flattens.axes() {
        l[axis] = m[axis];
        r[axis] = m[axis];
    }
    ([q[0], l, m], [m, r, q[2]])
}

/// `q` with its control point clamped, on each axis, between its ends.
/// Then its steps cannot have opposite signs.
///
/// This is a no-op on every piece a cut produced in exact arithmetic. What
/// it moves is rounding: a residual step of an ulp's size, or the
/// overshoot a cut too close to an end would have removed. That is the
/// geometry the kernel's certificate draws, applied on the host, so what
/// the table stores and what the kernel assumes are the same arc.
fn certified([p0, mut p1, p2]: [P; 3]) -> [P; 3] {
    for ((control, start), end) in p1.iter_mut().zip(p0).zip(p2) {
        *control = control.clamp(start.min(end), start.max(end));
    }
    [p0, p1, p2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn bezier([p0, p1, p2]: [P; 3], t: f64) -> P {
        let (a, b, c) = ((1.0 - t) * (1.0 - t), 2.0 * t * (1.0 - t), t * t);
        [
            a * p0[0] + b * p1[0] + c * p2[0],
            a * p0[1] + b * p1[1] + c * p2[1],
        ]
    }

    fn is_monotone(q: MonotoneQuad) -> bool {
        let [p0, p1, p2] = q.points();
        (0..2).all(|axis| {
            let (s0, s1) = (p1[axis] - p0[axis], p2[axis] - p1[axis]);
            !((s0 < 0.0 && s1 > 0.0) || (s0 > 0.0 && s1 < 0.0))
        })
    }

    /// The distance from `p` to the arc `q`: the least distance over a set of
    /// candidate points *on the arc*. Every candidate is a point of the arc,
    /// so the minimum can never undershoot the true distance. Including the
    /// true nearest point is what makes it exact.
    ///
    /// The nearest point of a quadratic is at an end or at a real root of
    /// the cubic `(B(t) − p)·B′(t) = 0`. So the candidates are:
    /// - both ends;
    /// - the cubic's roots in `[0, 1]`, solved in closed form and polished
    ///   by Newton;
    /// - dense samples, each refined by a golden-section search around it.
    ///
    /// The roots are what reach the right branch when the arc retraces
    /// itself (a hook, a cusp). A near-cusp's hairpin tail can be shorter
    /// than one sample interval, so the samples alone bracket both
    /// branches. Golden-section search then settles on either one: CI
    /// measured 2.7e-9 for a piece that lies within 3e-12 of the arc, which
    /// is exactly the width of the hairpin between its branches.
    fn distance_to_arc(q: [P; 3], p: P) -> f64 {
        const SAMPLES: usize = 1024;
        let dist = |t: f64| {
            let b = bezier(q, t);
            (b[0] - p[0]).hypot(b[1] - p[1])
        };
        let at = |k: usize| k as f64 / SAMPLES as f64;
        let sampled: Vec<f64> = (0..=SAMPLES).map(|k| dist(at(k))).collect();
        let phi = 0.5 * (5f64.sqrt() - 1.0);
        let refined = (0..=SAMPLES)
            .filter(|&k| {
                (k == 0 || sampled[k] <= sampled[k - 1])
                    && (k == SAMPLES || sampled[k] <= sampled[k + 1])
            })
            .map(|k| {
                let (mut lo, mut hi) = (at(k.saturating_sub(1)), at((k + 1).min(SAMPLES)));
                for _ in 0..80 {
                    let (a, b) = (hi - phi * (hi - lo), lo + phi * (hi - lo));
                    match dist(a) < dist(b) {
                        true => hi = b,
                        false => lo = a,
                    }
                }
                dist(0.5 * (lo + hi)).min(sampled[k])
            })
            .fold(f64::INFINITY, f64::min);
        nearest_parameters(q, p)
            .into_iter()
            .map(dist)
            .fold(refined, f64::min)
    }

    /// The parameters in `[0, 1]` where `|B(t) − p|` can be least: both ends,
    /// and every real root of `f(t) = (B(t) − p)·B′(t)`, a cubic. Each root is
    /// found in closed form, then polished by Newton on `f`.
    ///
    /// With `B(t) = p0 + 2t·a + t²·c`, where `a = p1 − p0` and
    /// `c = p2 − 2p1 + p0`, and `d = p0 − p`:
    /// `f(t)/2 = (c·c)t³ + 3(a·c)t² + (2a·a + d·c)t + d·a`.
    fn nearest_parameters([p0, p1, p2]: [P; 3], p: P) -> Vec<f64> {
        let dot = |u: P, v: P| u[0] * v[0] + u[1] * v[1];
        let a = [p1[0] - p0[0], p1[1] - p0[1]];
        let c = [p2[0] - 2.0 * p1[0] + p0[0], p2[1] - 2.0 * p1[1] + p0[1]];
        let d = [p0[0] - p[0], p0[1] - p[1]];
        let coeffs = [
            dot(c, c),
            3.0 * dot(a, c),
            2.0 * dot(a, a) + dot(d, c),
            dot(d, a),
        ];
        let f = |t: f64| ((coeffs[0] * t + coeffs[1]) * t + coeffs[2]) * t + coeffs[3];
        let df = |t: f64| (3.0 * coeffs[0] * t + 2.0 * coeffs[1]) * t + coeffs[2];
        let polish = |mut t: f64| {
            for _ in 0..8 {
                let slope = df(t);
                if slope == 0.0 {
                    break;
                }
                t = (t - f(t) / slope).clamp(0.0, 1.0);
            }
            t
        };
        let mut ts = vec![0.0, 1.0];
        ts.extend(
            real_cubic_roots(coeffs)
                .into_iter()
                .map(|t| polish(t.clamp(0.0, 1.0))),
        );
        ts
    }

    /// The real roots of `k0·t³ + k1·t² + k2·t + k3`, in closed form:
    /// Cardano when one root is real, the trigonometric form when three are.
    /// It falls back to the quadratic (or linear) formula when the leading
    /// coefficient vanishes, as it does for a straight arc.
    fn real_cubic_roots([k0, k1, k2, k3]: [f64; 4]) -> Vec<f64> {
        if k0 == 0.0 {
            if k1 == 0.0 {
                return if k2 == 0.0 { vec![] } else { vec![-k3 / k2] };
            }
            let disc = k2 * k2 - 4.0 * k1 * k3;
            if disc < 0.0 {
                return vec![];
            }
            let s = disc.sqrt();
            return vec![(-k2 + s) / (2.0 * k1), (-k2 - s) / (2.0 * k1)];
        }
        let (b, c, e) = (k1 / k0, k2 / k0, k3 / k0);
        // t = u − b/3 turns it into u³ + pu + q.
        let shift = b / 3.0;
        let p = c - b * b / 3.0;
        let q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + e;
        let disc = (q / 2.0) * (q / 2.0) + (p / 3.0) * (p / 3.0) * (p / 3.0);
        if disc > 0.0 {
            let s = disc.sqrt();
            let u = (-q / 2.0 + s).cbrt() + (-q / 2.0 - s).cbrt();
            return vec![u - shift];
        }
        if p == 0.0 {
            return vec![-shift];
        }
        let r = 2.0 * (-p / 3.0).sqrt();
        let phi = (3.0 * q / (p * r)).clamp(-1.0, 1.0).acos() / 3.0;
        (0..3)
            .map(|k| r * (phi - 2.0 * std::f64::consts::PI * f64::from(k) / 3.0).cos() - shift)
            .collect()
    }

    /// Every property the split promises: pieces in order, joined bit for
    /// bit, from `p0` to `p2`; each monotone; each tracing the original
    /// arc to within `tolerance` times the arc's size.
    fn assert_splits_faithfully(q: [P; 3], tolerance: f64) -> Vec<MonotoneQuad> {
        let pieces = MonotoneQuad::split(q[0], q[1], q[2]);
        assert!((1..=3).contains(&pieces.len()), "{} pieces", pieces.len());
        assert_eq!(pieces[0].points()[0], q[0], "the first piece starts at p0");
        assert_eq!(
            pieces[pieces.len() - 1].points()[2],
            q[2],
            "the last ends at p2"
        );
        for pair in pieces.windows(2) {
            assert_eq!(
                pair[0].points()[2],
                pair[1].points()[0],
                "a gap between pieces"
            );
        }
        let size = q
            .iter()
            .flat_map(|a| q.iter().map(move |b| (a[0] - b[0]).hypot(a[1] - b[1])))
            .fold(0.0, f64::max);
        for &piece in &pieces {
            assert!(is_monotone(piece), "{piece:?} turns back");
            for k in 0..=16 {
                let p = bezier(piece.points(), f64::from(k) / 16.0);
                let off = distance_to_arc(q, p);
                assert!(
                    off <= tolerance * (1.0 + size),
                    "{piece:?} strays {off} from the arc at s = {k}/16"
                );
            }
        }
        pieces
    }

    #[test]
    fn a_monotone_quadratic_is_not_cut() {
        let q = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let pieces = MonotoneQuad::split(q[0], q[1], q[2]);
        assert_eq!(pieces, vec![MonotoneQuad(q)], "cut or moved, bit for bit");
    }

    /// The hook the glyph's old midpoint split was tested on: it runs 50
    /// units left before returning, and rises by a quarter of 0.005 on the
    /// way. One turning point per axis, so three pieces, cut at
    /// `x = −10000/201` and at `y = 0.0025`.
    #[test]
    fn the_hook_is_cut_where_it_turns_on_each_axis() {
        let q = [[0.0, 0.0], [-100.0, 0.005], [1.0, 0.0]];
        let pieces = assert_splits_faithfully(q, 1e-9);
        assert_eq!(pieces.len(), 3);
        let ends: Vec<P> = pieces.iter().map(|p| p.points()[2]).collect();
        assert!(
            (ends[0][0] - (-10000.0 / 201.0)).abs() < 1e-9,
            "x turns at {ends:?}"
        );
        assert!((ends[1][1] - 0.0025).abs() < 1e-12, "y turns at {ends:?}");
    }

    /// A monotone arc under a compound component's rotation is not monotone
    /// any more — this is what a 2×2 component transform does to a font's
    /// curves, and why the split exists at all.
    #[test]
    fn a_rotated_arc_is_cut_where_rotation_made_it_turn() {
        let (c, s) = (30f64.to_radians().cos(), 30f64.to_radians().sin());
        let rotate = |[x, y]: P| [c * x - s * y, s * x + c * y];
        let q = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]].map(rotate);
        let pieces = assert_splits_faithfully(q, 1e-9);
        assert_eq!(pieces.len(), 2, "turns in x only: {pieces:?}");
    }

    /// Step 5's counterexample: the `x` and `y` turning points one ulp
    /// apart. A split that re-tested its own pieces never finished; this one
    /// makes one cut, a cusp, and stops.
    #[test]
    fn a_near_cusp_is_one_cut_and_terminates() {
        let q = [
            [3.12639927602814, -3.695140569535391],
            [-0.46210203107480297, 0.27566182495800406],
            [0.25748132579460403, -0.5205823650159438],
        ];
        let (tx, ty) = (turning_point(q, 0), turning_point(q, 1));
        assert!(
            matches!((tx, ty), (Some(a), Some(b)) if a != b && (a - b).abs() < 1e-15),
            "the input is the near-cusp it is named for: {tx:?} {ty:?}"
        );
        let pieces = assert_splits_faithfully(q, 1e-9);
        assert_eq!(pieces.len(), 2, "one cusp, one cut: {pieces:?}");
    }

    /// CI's proptest counterexample for `near_cusps_split_into_monotone_pieces_of_themselves`:
    /// a collinear arc overshooting its end by 0.05%, with `f32`-rounded
    /// points. The rounding leaves its hairpin tail about 3e-9 wide, and the
    /// tail is shorter than one of the distance oracle's sample intervals.
    /// The split is faithful: the last piece lies within 3e-12 of the arc.
    /// The oracle, not the split, is what reported 2.7e-9, when it sampled
    /// without the cubic's roots.
    #[test]
    fn a_hairpin_shorter_than_a_sample_is_measured_on_its_own_branch() {
        let q = [
            [f64::from(-39.539_246_f32), f64::from(52.064_91_f32)],
            [f64::from(-38.004_406_f32), f64::from(52.515_488_f32)],
            [f64::from(-38.005_245_f32), f64::from(52.515_24_f32)],
        ];
        let pieces = assert_splits_faithfully(q, 1e-9);
        assert_eq!(pieces.len(), 3, "an x and a y turning point: {pieces:?}");
    }

    #[test]
    fn degenerate_and_collinear_quadratics() {
        // A point: nothing turns.
        let point = [[2.0, 3.0]; 3];
        assert_eq!(MonotoneQuad::split(point[0], point[1], point[2]).len(), 1);
        // A straight arc with its control between the ends: nothing turns.
        let between = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        assert_eq!(assert_splits_faithfully(between, 1e-12).len(), 1);
        // The control beyond the far end: the arc overshoots to 2.25 and
        // comes back — a cusp on both axes at once, so two straight pieces.
        let beyond = [[0.0, 0.0], [3.0, 3.0], [2.0, 2.0]];
        let pieces = assert_splits_faithfully(beyond, 1e-12);
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces[0].points()[2], [2.25, 2.25]);
        // The same on one axis only: `y` never moves, `x` overshoots.
        let flat = [[0.0, 0.0], [3.0, 0.0], [2.0, 0.0]];
        assert_eq!(assert_splits_faithfully(flat, 1e-12).len(), 2);
        // A closed loop: its ends coincide, it turns at t = ½ on both axes
        // and retraces itself. (A caller drops this before splitting; see
        // the module docs. The split itself still keeps its promises.)
        let loop_ = [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]];
        assert_eq!(assert_splits_faithfully(loop_, 1e-12).len(), 2);
    }

    /// A turning point within [`SPLIT_MARGIN`] of an end is not cut; the
    /// certificate flattens the sliver of overshoot instead, and the arc
    /// still traces the original.
    #[test]
    fn a_turning_point_at_an_end_is_clamped_not_cut() {
        let q = [[0.0, 0.0], [-1e-12, 1.0], [1.0, 1.0]];
        assert!(turning_point(q, 0).is_some_and(|t| t < SPLIT_MARGIN));
        let pieces = assert_splits_faithfully(q, 1e-9);
        assert_eq!(pieces.len(), 1, "{pieces:?}");
        assert_eq!(
            pieces[0].points()[1],
            [0.0, 1.0],
            "the step is zeroed, not cut"
        );
    }

    fn coordinate() -> impl Strategy<Value = f64> {
        -100.0f64..100.0
    }

    fn point() -> impl Strategy<Value = P> {
        [coordinate(), coordinate()]
    }

    proptest! {
        /// Any quadratic.
        #[test]
        fn any_quadratic_splits_into_monotone_pieces_of_itself(q in [point(), point(), point()]) {
            assert_splits_faithfully(q, 1e-9);
        }

        /// Near-cusps: collinear quadratics whose control overshoots an end,
        /// nudged off the line — the family where the two turning points
        /// crowd together — in `f32`-representable coordinates, as glyph
        /// outlines are.
        #[test]
        fn near_cusps_split_into_monotone_pieces_of_themselves(
            a in point(),
            direction in 0.0f64..std::f64::consts::TAU,
            length in 0.1f64..100.0,
            overshoot in 1.0f64..3.0,
            nudge in -1e-6f64..1e-6,
        ) {
            let along = [direction.cos(), direction.sin()];
            let at = |k: f64, off: f64| [
                f64::from((a[0] + k * along[0] - off * along[1]) as f32),
                f64::from((a[1] + k * along[1] + off * along[0]) as f32),
            ];
            let q = [at(0.0, 0.0), at(overshoot * length, nudge), at(length, 0.0)];
            assert_splits_faithfully(q, 1e-9);
        }
    }
}
