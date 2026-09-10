//! A glyph as geometry: closed contours of line and quadratic segments, in
//! whatever frame the caller asked for.
//!
//! Parsing a TrueType glyph produces an [`Outline`]; a rasterizer consumes
//! one. Keeping the two apart is what lets every affine map in a font — a
//! compound glyph's component placement, the em-square scale, the screen
//! flip — be applied to control points on the host, so that a coverage
//! kernel is built in the frame it will be evaluated in and its constants
//! are that frame's numbers. (It used to be the other way round: segments
//! became kernels in a normalized unit square and every transform was a
//! coordinate warp on the finished kernel.) Quadratics are closed under
//! affine maps, so nothing is lost by transforming the control points.

/// A point in the outline's frame: `[x, y]`.
pub type Point = [f32; 2];

/// One piece of a contour, with on-curve endpoints.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Segment {
    /// A straight edge.
    Line {
        /// Where it starts.
        from: Point,
        /// Where it ends.
        to: Point,
    },
    /// A quadratic Bézier: `control` is the off-curve point.
    Quad {
        /// Where it starts.
        from: Point,
        /// The off-curve control point.
        control: Point,
        /// Where it ends.
        to: Point,
    },
}

impl Segment {
    /// The on-curve point the segment starts at.
    #[must_use]
    pub fn from(self) -> Point {
        match self {
            Self::Line { from, .. } | Self::Quad { from, .. } => from,
        }
    }

    /// The on-curve point the segment ends at.
    #[must_use]
    pub fn to(self) -> Point {
        match self {
            Self::Line { to, .. } | Self::Quad { to, .. } => to,
        }
    }

    /// Every control point, in order — the polygon the segment lies inside.
    #[must_use]
    pub fn control_points(self) -> Vec<Point> {
        match self {
            Self::Line { from, to } => vec![from, to],
            Self::Quad { from, control, to } => vec![from, control, to],
        }
    }

    /// The segment's control points pushed through `m`.
    #[must_use]
    pub fn transformed(self, m: Affine) -> Self {
        match self {
            Self::Line { from, to } => Self::Line {
                from: m.apply(from),
                to: m.apply(to),
            },
            Self::Quad { from, control, to } => Self::Quad {
                from: m.apply(from),
                control: m.apply(control),
                to: m.apply(to),
            },
        }
    }
}

/// Why [`Contour::new`] refused a segment list.
///
/// A contour is the closed chain a winding computation walks — see
/// [`Contour::new`] — so every producer of one, TrueType parsing included,
/// funnels its failures through these same two cases rather than each
/// inventing its own notion of "not a contour".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContourError {
    /// No segments at all: a boundary with no edges encloses nothing.
    Empty,
    /// Segment `at`'s [`Segment::to`] does not exactly equal the next
    /// segment's (`(at + 1) % len`) [`Segment::from`] — the chain never
    /// gets back to where it started.
    NotClosed {
        /// The segment whose `to` breaks the chain.
        at: usize,
    },
}

impl std::fmt::Display for ContourError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "a contour needs at least one segment"),
            Self::NotClosed { at } => {
                write!(f, "segment {at} does not end where the next segment begins")
            }
        }
    }
}

impl std::error::Error for ContourError {}

/// A closed contour: each segment starts where the previous one ends, and
/// the last ends where the first starts.
#[derive(Clone, Debug, PartialEq)]
pub struct Contour {
    segments: Vec<Segment>,
}

impl Contour {
    /// Build a contour from `segments`, checking the closure a winding
    /// computation ([`loop_blinn::glyph`](super::loop_blinn::glyph)) depends
    /// on: each segment's [`Segment::to`] must exactly equal the next
    /// segment's [`Segment::from`], and the last segment's `to` must exactly
    /// equal the first segment's `from`.
    ///
    /// "Exactly" is `f32` bit equality, not a tolerance. Every point a
    /// well-formed caller passes here comes from one source — the same
    /// value read or computed once and shared by the two segments that meet
    /// there ([`Self::from_truetype_points`] and the affine map in `ttf.rs`
    /// both build segments this way) — so a real gap is a construction bug
    /// a tolerance would hide rather than catch.
    ///
    /// An empty list is refused too: it encloses nothing, so it was never a
    /// contour, and this stops that state from being called one.
    ///
    /// # Errors
    ///
    /// [`ContourError::Empty`] for no segments, [`ContourError::NotClosed`]
    /// for a chain that does not close.
    pub fn new(segments: Vec<Segment>) -> Result<Self, ContourError> {
        let n = segments.len();
        if n == 0 {
            return Err(ContourError::Empty);
        }
        for (i, s) in segments.iter().enumerate() {
            let next = segments[(i + 1) % n];
            if s.to() != next.from() {
                return Err(ContourError::NotClosed { at: i });
            }
        }
        Ok(Self { segments })
    }

    /// The segments, in the direction the font draws them.
    #[must_use]
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// A contour from TrueType's point list — `(x, y, on_curve)` in order —
    /// with the format's implicit on-curve points made explicit: two
    /// consecutive off-curve points imply an on-curve point at their
    /// midpoint. The contour starts at its first on-curve point, so every
    /// segment has on-curve endpoints. `None` for an empty point list —
    /// there is no contour to build.
    #[must_use]
    pub fn from_truetype_points(points: &[(f32, f32, bool)]) -> Option<Self> {
        let expanded: Vec<(Point, bool)> = points
            .iter()
            .enumerate()
            .flat_map(|(i, &(x, y, on))| {
                let (nx, ny, next_on) = points[(i + 1) % points.len()];
                if !on && !next_on {
                    vec![([x, y], on), ([(x + nx) / 2.0, (y + ny) / 2.0], true)]
                } else {
                    vec![([x, y], on)]
                }
            })
            .collect();
        let start = expanded.iter().position(|p| p.1)?;
        let at = |j: usize| expanded[(start + j) % expanded.len()];
        let mut segments = Vec::with_capacity(expanded.len());
        let mut i = 0;
        while i < expanded.len() {
            let (from, _) = at(i);
            let (next, next_on) = at(i + 1);
            if next_on {
                segments.push(Segment::Line { from, to: next });
                i += 1;
            } else {
                let (to, _) = at(i + 2);
                segments.push(Segment::Quad {
                    from,
                    control: next,
                    to,
                });
                i += 2;
            }
        }
        // Every segment's endpoints are read from `expanded` by index
        // (`at(j)` for a shared `j`, never recomputed), so consecutive
        // segments always share one bit-identical `Point` and this cannot
        // fail for a non-empty `expanded` — which `start` having been found
        // already proves.
        Some(
            Self::new(segments).expect(
                "from_truetype_points shares endpoints by array index, so it always closes",
            ),
        )
    }
}

/// A glyph's geometry: every contour, in one frame.
///
/// Contours may overlap, self-intersect, nest, and run in either direction;
/// what an outline *means* is its winding number under the non-zero rule,
/// which is what every consumer computes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outline {
    /// The contours.
    pub contours: Vec<Contour>,
}

impl Outline {
    /// Whether there is nothing to draw. A [`Contour`] can never itself be
    /// empty ([`Contour::new`] refuses one), so an outline has nothing to
    /// draw exactly when it has no contours at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.contours.is_empty()
    }

    /// Every segment of every contour.
    pub fn segments(&self) -> impl Iterator<Item = Segment> + '_ {
        self.contours
            .iter()
            .flat_map(|c| c.segments().iter().copied())
    }

    /// The outline pushed through `m`.
    #[must_use]
    pub fn transformed(&self, m: Affine) -> Self {
        Self {
            contours: self
                .contours
                .iter()
                .map(|c| {
                    let segments = c.segments().iter().map(|s| s.transformed(m)).collect();
                    // `m.apply` is a deterministic function of its input
                    // bits, so the same shared endpoint that made `c` close
                    // maps to the same output on both sides and the result
                    // closes too.
                    Contour::new(segments).expect(
                        "an affine map preserves the exact equality a closed contour's \
                         shared endpoints held",
                    )
                })
                .collect(),
        }
    }

    /// The outline translated by `[dx, dy]`.
    #[must_use]
    pub fn translated(&self, [dx, dy]: Point) -> Self {
        self.transformed(Affine::translation(dx, dy))
    }

    /// Every contour of `other`, appended — the outline of a string is the
    /// outlines of its glyphs, placed.
    pub fn append(&mut self, other: Outline) {
        self.contours.extend(other.contours);
    }

    /// The box `[x0, y0, x1, y1]` containing every control point, which
    /// contains every curve. `None` for an empty outline.
    #[must_use]
    pub fn bounds(&self) -> Option<[f32; 4]> {
        let mut out = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        let mut any = false;
        for [x, y] in self.segments().flat_map(Segment::control_points) {
            any = true;
            out[0] = out[0].min(x);
            out[1] = out[1].min(y);
            out[2] = out[2].max(x);
            out[3] = out[3].max(y);
        }
        any.then_some(out)
    }
}

/// The forward affine map `x' = a·x + b·y + tx, y' = c·x + d·y + ty`, stored
/// as `[a, b, c, d, tx, ty]` — TrueType's component-transform layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine(pub [f32; 6]);

impl Affine {
    /// The map that changes nothing.
    pub const IDENTITY: Self = Self([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    /// A pure translation.
    #[must_use]
    pub fn translation(tx: f32, ty: f32) -> Self {
        Self([1.0, 0.0, 0.0, 1.0, tx, ty])
    }

    /// `m(p)`.
    #[must_use]
    pub fn apply(self, [x, y]: Point) -> Point {
        let [a, b, c, d, tx, ty] = self.0;
        [a * x + b * y + tx, c * x + d * y + ty]
    }

    /// The map that applies `self` first and then `outer`: `outer ∘ self`.
    #[must_use]
    pub fn then(self, outer: Self) -> Self {
        let [a, b, c, d, tx, ty] = self.0;
        let [oa, ob, oc, od, otx, oty] = outer.0;
        Self([
            oa * a + ob * c,
            oa * b + ob * d,
            oc * a + od * c,
            oc * b + od * d,
            oa * tx + ob * ty + otx,
            oc * tx + od * ty + oty,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every contour a font produces is closed, segment to segment: that is
    /// the property every winding computation depends on, and the implicit
    /// midpoint expansion is where it would break. [`Contour::new`] (which
    /// [`Contour::from_truetype_points`] itself routes through) already
    /// refuses anything that fails this, so a `Some` result is proof — this
    /// restates the check directly, in the caller's own terms, as the
    /// property this module exists to protect.
    fn assert_closed(contour: &Contour) {
        let n = contour.segments().len();
        for (i, s) in contour.segments().iter().enumerate() {
            let next = contour.segments()[(i + 1) % n];
            assert_eq!(
                s.to(),
                next.from(),
                "segment {i} does not meet segment {}",
                (i + 1) % n
            );
        }
    }

    #[test]
    fn all_on_curve_points_make_lines() {
        let c = Contour::from_truetype_points(&[
            (0.0, 0.0, true),
            (4.0, 0.0, true),
            (4.0, 4.0, true),
            (0.0, 4.0, true),
        ])
        .expect("four points make a contour");
        assert_eq!(c.segments().len(), 4);
        assert!(c
            .segments()
            .iter()
            .all(|s| matches!(s, Segment::Line { .. })));
        assert_closed(&c);
    }

    #[test]
    fn consecutive_off_curve_points_imply_a_midpoint() {
        // A circle-ish contour of four off-curve points and no on-curve ones:
        // four implied midpoints, four quadratics, starting at a midpoint.
        let c = Contour::from_truetype_points(&[
            (1.0, 0.0, false),
            (1.0, 1.0, false),
            (0.0, 1.0, false),
            (0.0, 0.0, false),
        ])
        .expect("four points make a contour");
        assert_eq!(c.segments().len(), 4);
        assert!(c
            .segments()
            .iter()
            .all(|s| matches!(s, Segment::Quad { .. })));
        assert_eq!(c.segments()[0].from(), [1.0, 0.5]);
        assert_closed(&c);
    }

    #[test]
    fn a_contour_starting_off_curve_is_rotated_to_an_on_curve_start() {
        let c =
            Contour::from_truetype_points(&[(0.0, 1.0, false), (1.0, 0.0, true), (2.0, 2.0, true)])
                .expect("three points make a contour");
        assert_eq!(c.segments().len(), 2);
        assert_eq!(c.segments()[0].from(), [1.0, 0.0]);
        assert!(matches!(c.segments()[1], Segment::Quad { .. }));
        assert_closed(&c);
    }

    #[test]
    fn a_single_off_curve_point_is_a_degenerate_quad() {
        let c = Contour::from_truetype_points(&[(3.0, 3.0, false)])
            .expect("one point makes a degenerate contour");
        assert_eq!(c.segments().len(), 1);
        assert_eq!(
            c.segments()[0],
            Segment::Quad {
                from: [3.0, 3.0],
                control: [3.0, 3.0],
                to: [3.0, 3.0]
            }
        );
    }

    #[test]
    fn an_empty_point_list_is_no_contour_at_all() {
        assert!(Contour::from_truetype_points(&[]).is_none());
    }

    /// The finding this module's constructor exists for: a single edge that
    /// never returns to its own start is not a boundary, and building one by
    /// hand must be refused rather than silently accepted as a "contour"
    /// that a winding computation would then misread as a filled half-plane.
    #[test]
    fn a_single_open_line_is_refused() {
        let err = Contour::new(vec![Segment::Line {
            from: [0.0, 0.0],
            to: [0.0, 10.0],
        }])
        .expect_err("a line's `to` does not equal its own `from`");
        assert_eq!(err, ContourError::NotClosed { at: 0 });
    }

    /// A chain with every join sound but one: refused at the joint that
    /// breaks, not merely "eventually" — [`ContourError::NotClosed`]'s `at`
    /// names the segment.
    #[test]
    fn a_chain_with_one_gap_is_refused() {
        let segments = vec![
            Segment::Line {
                from: [0.0, 0.0],
                to: [4.0, 0.0],
            },
            Segment::Line {
                from: [4.0, 0.0],
                to: [4.0, 4.0],
            },
            // Should return to [0.0, 0.0] to close the loop; instead it
            // stops short, leaving a gap only the last segment can see.
            Segment::Line {
                from: [4.0, 4.0],
                to: [0.0, 4.0],
            },
        ];
        let err = Contour::new(segments).expect_err("the loop never returns to [0.0, 0.0]");
        assert_eq!(err, ContourError::NotClosed { at: 2 });
    }

    #[test]
    fn an_empty_segment_list_is_refused() {
        assert_eq!(Contour::new(vec![]), Err(ContourError::Empty));
    }

    /// A closed square and a closed ring — the shapes
    /// `tests/loop_blinn_winding.rs` winds against an independent oracle —
    /// are accepted. This only pins acceptance of the constructor itself;
    /// the winding tests already cover what `loop_blinn::glyph` does with
    /// the result.
    #[test]
    fn a_closed_square_and_ring_are_accepted() {
        let square = Contour::new(
            (0..4)
                .map(|i| {
                    let pts = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
                    Segment::Line {
                        from: pts[i],
                        to: pts[(i + 1) % 4],
                    }
                })
                .collect(),
        );
        assert!(square.is_ok());

        let on = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];
        let off = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]];
        let ring = Contour::new(
            (0..4)
                .map(|i| Segment::Quad {
                    from: on[i],
                    control: off[i],
                    to: on[(i + 1) % 4],
                })
                .collect(),
        );
        assert!(ring.is_ok());
    }

    #[test]
    fn affine_composition_applies_left_to_right() {
        let scale = Affine([2.0, 0.0, 0.0, 3.0, 0.0, 0.0]);
        let shift = Affine::translation(1.0, -1.0);
        let p = [1.0, 1.0];
        assert_eq!(scale.then(shift).apply(p), shift.apply(scale.apply(p)));
        assert_eq!(scale.then(shift).apply(p), [3.0, 2.0]);
        assert_eq!(Affine::IDENTITY.then(scale), scale);
    }

    #[test]
    fn bounds_cover_control_points() {
        let mut o = Outline::default();
        // Closed by a straight edge back to the start; its own control
        // points ([1,1] and [0,0]) fall inside the quadratic's bounds, so
        // the expected box is unchanged by adding it.
        o.contours.push(
            Contour::new(vec![
                Segment::Quad {
                    from: [0.0, 0.0],
                    control: [5.0, -2.0],
                    to: [1.0, 1.0],
                },
                Segment::Line {
                    from: [1.0, 1.0],
                    to: [0.0, 0.0],
                },
            ])
            .expect("the line closes the quadratic's loop"),
        );
        assert_eq!(o.bounds(), Some([0.0, -2.0, 5.0, 1.0]));
        assert_eq!(Outline::default().bounds(), None);
        assert!(Outline::default().is_empty());
    }
}
