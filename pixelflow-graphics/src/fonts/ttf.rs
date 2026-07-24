//! pixelflow-graphics/src/fonts/ttf.rs
//!
//! TTF parser producing glyph coverage [`Kernel`]s.
//!
//! A glyph is ONE fused coverage kernel: outline segments become leaf kernels
//! ([`AnalyticalLine::kernel`] / [`AnalyticalQuad::kernel`]), the winding rule
//! is `sum(...).abs().min(1)`, bounds are a unit-square mask `select`, and
//! every transform (glyph restore, compound children, scaling) is a coordinate
//! warp via [`Kernel::at`]. Nothing here is a scene graph of Rust types — the
//! arena is the program, composed at parse time and compiled once at bake.
//!
//! Antialiasing comes from the leaf kernels' symbolic `Dwrt` ramps: the
//! derivatives chain through every `at` warp, so the crossing ramp is ~1
//! *screen* pixel wide at any glyph scale. No jet domain.

use pixelflow_core::Kernel;

// Import analytical curve leaf kernels
use super::ttf_curve_analytical::{AnalyticalLine, AnalyticalQuad};

// ═══════════════════════════════════════════════════════════════════════════
// Kernel composition helpers
// ═══════════════════════════════════════════════════════════════════════════

/// `Kernel::constant` shorthand for this module's builders.
fn kc(v: f32) -> Kernel {
    Kernel::constant(v)
}

/// The unit-square bounds mask `X≥0 & X≤1 & Y≥0 & Y≤1` — the glyph's
/// short-circuit bounds check as a Kernel mask.
fn unit_square_mask() -> Kernel {
    Kernel::x()
        .ge(&kc(0.0))
        .and(&Kernel::x().le(&kc(1.0)))
        .and(&Kernel::y().ge(&kc(0.0)))
        .and(&Kernel::y().le(&kc(1.0)))
}

/// Sample `inner` through the affine transform `[a, b, c, d, tx, ty]` — the
/// forward map `x' = a·x + b·y + tx, y' = c·x + d·y + ty` — by warping
/// coordinates with the INVERSE matrix:
/// `u = (X - tx)·inv_a + (Y - ty)·inv_b`, `v = (X - tx)·inv_c + (Y - ty)·inv_d`.
pub(crate) fn affine_kernel(inner: &Kernel, [a, b, c, d, tx, ty]: [f32; 6]) -> Kernel {
    let det = a * d - b * c;
    let inv_det = if det.abs() < 1e-6 { 0.0 } else { 1.0 / det };

    let inv_a = d * inv_det;
    let inv_b = -b * inv_det;
    let inv_c = -c * inv_det;
    let inv_d = a * inv_det;

    let coord = |ca: f32, cb: f32| {
        Kernel::x()
            .sub(&kc(tx))
            .mul(&kc(ca))
            .add(&Kernel::y().sub(&kc(ty)).mul(&kc(cb)))
    };
    inner.at(
        &coord(inv_a, inv_b),
        &coord(inv_c, inv_d),
        &Kernel::z(),
        &Kernel::w(),
    )
}

/// Winding coverage for a set of segment kernels: `min(|Σ|, 1)` — the
/// non-zero fill rule over the summed line/quad contributions.
fn coverage(segments: &[Kernel]) -> Kernel {
    Kernel::sum(segments).abs().min(&kc(1.0))
}

// ═══════════════════════════════════════════════════════════════════════════
// Reader
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
struct R<'a>(&'a [u8], usize);

impl<'a> R<'a> {
    fn u8(&mut self) -> Option<u8> {
        let v = *self.0.get(self.1)?;
        self.1 += 1;
        Some(v)
    }
    fn i8(&mut self) -> Option<i8> {
        self.u8().map(|v| v as i8)
    }
    fn u16(&mut self) -> Option<u16> {
        let s = self.0.get(self.1..self.1 + 2)?;
        self.1 += 2;
        Some(u16::from_be_bytes(s.try_into().ok()?))
    }
    fn i16(&mut self) -> Option<i16> {
        self.u16().map(|v| v as i16)
    }
    fn u32(&mut self) -> Option<u32> {
        let s = self.0.get(self.1..self.1 + 4)?;
        self.1 += 4;
        Some(u32::from_be_bytes(s.try_into().ok()?))
    }
    fn skip(&mut self, n: usize) -> Option<()> {
        self.0.get(self.1..self.1 + n)?;
        self.1 += n;
        Some(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tables (Dependent Types)
// ═══════════════════════════════════════════════════════════════════════════

enum Loca<'a> {
    Short(&'a [u8]),
    Long(&'a [u8]),
}

impl Loca<'_> {
    fn get(&self, i: usize) -> Option<usize> {
        match self {
            Self::Short(d) => Some(R(d, i * 2).u16()? as usize * 2),
            Self::Long(d) => Some(R(d, i * 4).u32()? as usize),
        }
    }
}

enum Cmap<'a> {
    Fmt4(&'a [u8]),
    Fmt12(&'a [u8]),
}

impl Cmap<'_> {
    fn lookup(&self, c: u32) -> Option<u16> {
        match self {
            Self::Fmt4(d) if c <= 0xFFFF => {
                let n = R(d, 6).u16()? as usize / 2;
                (0..n).find_map(|i| {
                    let end = R(d, 14 + i * 2).u16()?;
                    if c as u16 > end {
                        return None;
                    }
                    let start = R(d, 16 + n * 2 + i * 2).u16()?;
                    if (c as u16) < start {
                        return Some(0);
                    }
                    let delta = R(d, 16 + n * 4 + i * 2).i16()?;
                    let range = R(d, 16 + n * 6 + i * 2).u16()?;
                    Some(if range == 0 {
                        (c as i16).wrapping_add(delta) as u16
                    } else {
                        let off =
                            16 + n * 6 + i * 2 + range as usize + (c as u16 - start) as usize * 2;
                        let g = R(d, off).u16()?;
                        if g == 0 {
                            0
                        } else {
                            (g as i16).wrapping_add(delta) as u16
                        }
                    })
                })
            }
            Self::Fmt12(d) => (0..R(d, 12).u32()? as usize).find_map(|i| {
                let (s, e, g) = (
                    R(d, 16 + i * 12).u32()?,
                    R(d, 20 + i * 12).u32()?,
                    R(d, 24 + i * 12).u32()?,
                );
                (c >= s && c <= e).then(|| (g + c - s) as u16)
            }),
            _ => None,
        }
    }
}

enum Kern<'a> {
    /// Format 0: sorted pairs (left_glyph, right_glyph, value)
    Fmt0 { data: &'a [u8], n_pairs: usize },
    /// No kerning table
    None,
}

impl<'a> Kern<'a> {
    fn parse(data: &'a [u8]) -> Self {
        let Some(n_tables) = R(data, 2).u16() else {
            return Self::None;
        };
        let mut off = 4;

        for _ in 0..n_tables {
            let Some(length) = R(data, off + 2).u16() else {
                return Self::None;
            };
            let Some(coverage) = R(data, off + 4).u16() else {
                return Self::None;
            };

            let format = coverage >> 8;
            let horizontal = coverage & 1;

            if format == 0 && horizontal == 1 {
                let Some(n_pairs) = R(data, off + 6).u16() else {
                    return Self::None;
                };
                return Self::Fmt0 {
                    data: &data[off + 14..], // Skip header to pairs
                    n_pairs: n_pairs as usize,
                };
            }
            off += length as usize;
        }
        Self::None
    }

    fn get(&self, left: u16, right: u16) -> i16 {
        match self {
            Self::Fmt0 { data, n_pairs } => {
                // Binary search: each pair is 6 bytes (left:2, right:2, value:2)
                let key = ((left as u32) << 16) | (right as u32);
                let (mut lo, mut hi) = (0, *n_pairs);

                while lo < hi {
                    let mid = (lo + hi) / 2;
                    let pair = ((R(data, mid * 6).u16().unwrap_or(0) as u32) << 16)
                        | (R(data, mid * 6 + 2).u16().unwrap_or(0) as u32);

                    match pair.cmp(&key) {
                        std::cmp::Ordering::Less => lo = mid + 1,
                        std::cmp::Ordering::Greater => hi = mid,
                        std::cmp::Ordering::Equal => {
                            return R(data, mid * 6 + 4).i16().unwrap_or(0);
                        }
                    }
                }
                0
            }
            Self::None => 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Font
// ═══════════════════════════════════════════════════════════════════════════

// TTF/OTF Platform IDs
const PLATFORM_UNICODE: u16 = 0;
const PLATFORM_WINDOWS: u16 = 3;

// TTF/OTF Encoding IDs
const ENCODING_WINDOWS_UNICODE_BMP: u16 = 1;
const ENCODING_WINDOWS_UNICODE_FULL: u16 = 10;
const ENCODING_UNICODE_2_0_BMP: u16 = 3;
const ENCODING_UNICODE_2_0_FULL: u16 = 4;

// TTF/OTF Format IDs
const FORMAT_SEGMENT_MAPPING: u16 = 4;
const FORMAT_SEGMENTED_COVERAGE: u16 = 12;

/// Normalization parameters for simple glyphs.
struct Normalization {
    scale: f32,
    tx: f32,
    ty: f32,
}

pub struct Font<'a> {
    data: &'a [u8],
    glyf: usize,
    loca: Loca<'a>,
    cmap: Cmap<'a>,
    kern: Kern<'a>,
    hmtx: usize,
    num_hm: usize,
    pub units_per_em: u16,
    pub ascent: i16,
    pub descent: i16,
    pub line_gap: i16,
}

impl<'a> Font<'a> {
    #[must_use]
    pub fn parse(data: &'a [u8]) -> Option<Self> {
        // TTF header: sfntVersion(4) + numTables(2) + searchRange(2) + entrySelector(2) + rangeShift(2) = 12 bytes
        // Table record: tag(4) + checksum(4) + offset(4) + length(4) = 16 bytes
        let num_tables = R(data, 4).u16()? as usize;
        let mut t = std::collections::HashMap::new();

        for i in 0..num_tables {
            let rec = 12 + i * 16;
            let tag = [data[rec], data[rec + 1], data[rec + 2], data[rec + 3]];
            let offset = R(data, rec + 8).u32()? as usize;
            t.insert(tag, offset);
        }

        let head = *t.get(b"head")?;
        let loca = *t.get(b"loca")?;
        let hhea = *t.get(b"hhea")?;

        Some(Self {
            data,
            glyf: *t.get(b"glyf")?,
            loca: if R(data, head + 50).i16()? != 0 {
                Loca::Long(&data[loca..])
            } else {
                Loca::Short(&data[loca..])
            },
            cmap: Self::find_cmap(&data[*t.get(b"cmap")?..])?,
            kern: t
                .get(b"kern")
                .map(|&off| Kern::parse(&data[off..]))
                .unwrap_or(Kern::None),
            hmtx: *t.get(b"hmtx")?,
            num_hm: R(data, hhea + 34).u16()? as usize,
            units_per_em: R(data, head + 18).u16()?,
            ascent: R(data, hhea + 4).i16()?,
            descent: R(data, hhea + 6).i16()?,
            line_gap: R(data, hhea + 8).i16()?,
        })
    }

    fn find_cmap(d: &'a [u8]) -> Option<Cmap<'a>> {
        (0..R(d, 2).u16()? as usize)
            .filter_map(|i| {
                let (p, e, o) = (
                    R(d, 4 + i * 8).u16()?,
                    R(d, 6 + i * 8).u16()?,
                    R(d, 8 + i * 8).u32()? as usize,
                );
                let f = R(d, o).u16()?;
                match (p, e, f) {
                    (
                        PLATFORM_WINDOWS,
                        ENCODING_WINDOWS_UNICODE_FULL,
                        FORMAT_SEGMENTED_COVERAGE,
                    )
                    | (PLATFORM_UNICODE, ENCODING_UNICODE_2_0_FULL, FORMAT_SEGMENTED_COVERAGE) => {
                        Some((2, o, f))
                    }

                    (PLATFORM_WINDOWS, ENCODING_WINDOWS_UNICODE_BMP, FORMAT_SEGMENT_MAPPING)
                    | (PLATFORM_UNICODE, ENCODING_UNICODE_2_0_BMP, FORMAT_SEGMENT_MAPPING) => {
                        Some((1, o, f))
                    }
                    _ => None,
                }
            })
            .max_by_key(|x| x.0)
            .and_then(|(_, o, f)| match f {
                4 => Some(Cmap::Fmt4(&d[o..])),
                12 => Some(Cmap::Fmt12(&d[o..])),
                _ => None,
            })
    }

    /// Lookup a glyph ID from a codepoint (single CMAP lookup).
    ///
    /// Use this when you need the glyph ID to batch multiple operations,
    /// avoiding redundant CMAP lookups in tight loops.
    #[inline]
    #[must_use]
    pub fn cmap_lookup(&self, ch: char) -> Option<u16> {
        self.cmap.lookup(ch as u32)
    }

    /// The glyph for `ch` as a coverage [`Kernel`] in font units. Bake it once
    /// (`Lattice::bake`) or compose it into a scene; antialiasing resolves
    /// from `Dwrt` at bake.
    #[must_use]
    pub fn glyph_kernel(&self, ch: char) -> Option<Kernel> {
        self.compile(self.cmap.lookup(ch as u32)?)
    }

    /// [`Font::glyph_kernel`] by pre-looked-up glyph ID.
    #[must_use]
    pub fn glyph_kernel_by_id(&self, id: u16) -> Option<Kernel> {
        self.compile(id)
    }

    /// The `size`-scaled glyph for `ch` as a coverage [`Kernel`]: the ascent
    /// line sits at screen y=0 (top) and the descent at y=`size`, with
    /// screen Y increasing downward.
    #[must_use]
    pub fn glyph_kernel_scaled(&self, ch: char, size: f32) -> Option<Kernel> {
        let id = self.cmap.lookup(ch as u32)?;
        self.glyph_kernel_scaled_by_id(id, size)
    }

    /// [`Font::glyph_kernel_scaled`] by pre-looked-up glyph ID.
    #[must_use]
    pub fn glyph_kernel_scaled_by_id(&self, id: u16, size: f32) -> Option<Kernel> {
        let g = self.compile(id)?;
        // Scale based on total font height (ascent + |descent|) to fit within
        // `size` pixels; Y-flip because screen Y increases downward while font
        // Y increases upward. Forward: screen = [scale, 0, 0, -scale] · font
        // + (0, ascent_px).
        let total_height = self.ascent as f32 + self.descent.abs() as f32;
        let scale = size / total_height;
        let ascent_px = self.ascent as f32 * scale;
        Some(affine_kernel(&g, [scale, 0.0, 0.0, -scale, 0.0, ascent_px]))
    }

    #[must_use]
    pub fn advance(&self, ch: char) -> Option<f32> {
        let id = self.cmap.lookup(ch as u32)?;
        self.advance_by_id(id)
    }

    /// Get advance width in font units by pre-looked-up glyph ID.
    ///
    /// Avoids redundant CMAP lookup when you already have the glyph ID.
    #[inline]
    #[must_use]
    pub fn advance_by_id(&self, id: u16) -> Option<f32> {
        let i = (id as usize).min(self.num_hm.saturating_sub(1));
        Some(R(self.data, self.hmtx + i * 4).u16()? as f32)
    }

    #[must_use]
    pub fn advance_scaled(&self, ch: char, size: f32) -> Option<f32> {
        Some(self.advance(ch)? * size / self.units_per_em as f32)
    }

    /// Get scaled advance width by pre-looked-up glyph ID.
    ///
    /// Avoids redundant CMAP lookup when you already have the glyph ID.
    #[must_use]
    pub fn advance_scaled_by_id(&self, id: u16, size: f32) -> Option<f32> {
        Some(self.advance_by_id(id)? * size / self.units_per_em as f32)
    }

    /// Get kerning adjustment between two characters in font units.
    #[must_use]
    pub fn kern(&self, left: char, right: char) -> f32 {
        let left_id = self.cmap.lookup(left as u32).unwrap_or(0);
        let right_id = self.cmap.lookup(right as u32).unwrap_or(0);
        self.kern_by_ids(left_id, right_id)
    }

    /// Get kerning adjustment between two pre-looked-up glyph IDs in font units.
    ///
    /// Avoids redundant CMAP lookups when you already have both glyph IDs.
    #[inline]
    #[must_use]
    pub fn kern_by_ids(&self, left_id: u16, right_id: u16) -> f32 {
        self.kern.get(left_id, right_id) as f32
    }

    /// Get kerning adjustment between two characters, scaled to size.
    #[must_use]
    pub fn kern_scaled(&self, left: char, right: char, size: f32) -> f32 {
        self.kern(left, right) * size / self.units_per_em as f32
    }

    /// Compile a glyph to its coverage [`Kernel`] in font units.
    ///
    /// Simple glyphs: parse segments in normalized [0,1] space, apply the
    /// winding rule + unit-square bounds, then warp back to font units.
    /// Compound glyphs: recursively compile children and sum them through
    /// their affine transforms. Empty glyphs are the constant 0.
    fn compile(&self, id: u16) -> Option<Kernel> {
        let (a, b) = (self.loca.get(id as usize)?, self.loca.get(id as usize + 1)?);
        if a == b {
            return Some(kc(0.0));
        }
        let mut r = R(self.data, self.glyf + a);
        let n = r.i16()?;
        let x_min = r.i16()?;
        let y_min = r.i16()?;
        let x_max = r.i16()?;
        let y_max = r.i16()?;

        let width = (x_max - x_min) as f32;
        let height = (y_max - y_min) as f32;
        let max_dim = width.max(height).max(1.0); // Avoid div by 0

        // Normalize transform: map [x_min, x_min+max_dim] -> [0, 1]
        let norm_scale = 1.0 / max_dim;
        let norm_tx = -(x_min as f32) * norm_scale;
        let norm_ty = -(y_min as f32) * norm_scale;

        // The restore transform maps [0, 1] back to font units
        // x_world = x_local * max_dim + x_min
        // y_world = y_local * max_dim + y_min (no flip - keep Y-down from TrueType)
        let restore = [max_dim, 0.0, 0.0, max_dim, x_min as f32, y_min as f32];

        if n >= 0 {
            // Parse segments in normalized [0,1] space
            let normalization = Normalization {
                scale: norm_scale,
                tx: norm_tx,
                ty: norm_ty,
            };
            let segments = self.simple(&mut r, n as usize, normalization)?;

            // Compose: winding coverage -> unit-square bounds -> font units.
            let bounded = unit_square_mask().select(&coverage(&segments), &kc(0.0));
            Some(affine_kernel(&bounded, restore))
        } else {
            // Compound glyphs: children are already fully composed with their own bounds
            self.compound(&mut r)
        }
    }

    /// Parse a simple glyph's outline into per-segment winding kernels in
    /// normalized [0,1] space.
    fn simple(&self, r: &mut R, n: usize, norm: Normalization) -> Option<Vec<Kernel>> {
        if n == 0 {
            return Some(Vec::new());
        }
        let ends: Vec<_> = (0..n)
            .map(|_| r.u16().map(|v| v as usize))
            .collect::<Option<_>>()?;
        let np = *ends.last()? + 1;
        let instr_len = r.u16()? as usize;
        r.skip(instr_len)?;

        let mut fl = Vec::with_capacity(np);
        while fl.len() < np {
            let f = r.u8()?;
            fl.push(f);
            if f & 8 != 0 {
                for _ in 0..r.u8()?.min((np - fl.len()) as u8) {
                    fl.push(f);
                }
            }
        }

        let dec = |r: &mut R, s: u8, m: u8| {
            fl.iter()
                .try_fold((0i16, vec![]), |(mut v, mut out), &f| {
                    v += match (f & s != 0, f & m != 0) {
                        (true, true) => r.u8()? as i16,
                        (true, false) => -(r.u8()? as i16),
                        (false, true) => 0,
                        (false, false) => r.i16()?,
                    };
                    out.push(v);
                    Some((v, out))
                })
                .map(|(_, v)| v)
        };

        let (xs, ys) = (dec(r, 2, 16)?, dec(r, 4, 32)?);

        // Normalize points immediately
        let pts: Vec<_> = (0..np)
            .map(|i| {
                (
                    (xs[i] as f32) * norm.scale + norm.tx,
                    (ys[i] as f32) * norm.scale + norm.ty,
                    fl[i] & 1 != 0,
                )
            })
            .collect();

        // Each contour contributes line/quad segment kernels.
        let mut segments = Vec::new();
        let mut start = 0;
        for &e in ends.iter() {
            let c = &pts[start..=e];
            start = e + 1;
            push_segs(c, &mut segments);
        }

        Some(segments)
    }

    fn compound(&self, r: &mut R) -> Option<Kernel> {
        let mut kids = vec![];
        loop {
            let fl = r.u16()?;
            let id = r.u16()?;
            let (dx, dy) = if fl & 2 != 0 {
                if fl & 1 != 0 {
                    (r.i16()?, r.i16()?)
                } else {
                    (r.i8()? as i16, r.i8()? as i16)
                }
            } else {
                r.skip(if fl & 1 != 0 { 4 } else { 2 })?;
                (0, 0)
            };
            let mut m = [1.0, 0.0, 0.0, 1.0, dx as f32, dy as f32];
            if fl & 0x08 != 0 {
                let s = r.i16()? as f32 / 16384.0;
                m[0] = s;
                m[3] = s;
            } else if fl & 0x40 != 0 {
                m[0] = r.i16()? as f32 / 16384.0;
                m[3] = r.i16()? as f32 / 16384.0;
            } else if fl & 0x80 != 0 {
                m[0] = r.i16()? as f32 / 16384.0;
                m[1] = r.i16()? as f32 / 16384.0;
                m[2] = r.i16()? as f32 / 16384.0;
                m[3] = r.i16()? as f32 / 16384.0;
            }
            if let Some(g) = self.compile(id) {
                kids.push(affine_kernel(&g, m));
            }
            if fl & 0x20 == 0 {
                break;
            }
        }
        Some(Kernel::sum(&kids))
    }
}

/// Convert one contour's point list into line/quad winding kernels.
fn push_segs(pts: &[(f32, f32, bool)], segments: &mut Vec<Kernel>) {
    if pts.is_empty() {
        return;
    }
    let exp: Vec<_> = pts
        .iter()
        .enumerate()
        .flat_map(|(i, &(x, y, on))| {
            let (nx, ny, non) = pts[(i + 1) % pts.len()];
            if !on && !non {
                vec![(x, y, on), ((x + nx) / 2.0, (y + ny) / 2.0, true)]
            } else {
                vec![(x, y, on)]
            }
        })
        .collect();

    if exp.is_empty() {
        return;
    }

    let start = exp.iter().position(|p| p.2).unwrap_or(0);
    let mut i = 0;
    while i < exp.len() {
        let p = |j: usize| {
            let (x, y, _) = exp[(start + j) % exp.len()];
            [x, y]
        };
        if exp[(start + i + 1) % exp.len()].2 {
            if let Some(line) = AnalyticalLine::from_points(p(i), p(i + 1)) {
                segments.push(line.kernel());
            }
            i += 1;
        } else {
            segments.push(AnalyticalQuad::new(p(i), p(i + 1), p(i + 2)).kernel());
            i += 2;
        }
    }
}
