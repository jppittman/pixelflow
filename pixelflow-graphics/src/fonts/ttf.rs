//! pixelflow-graphics/src/fonts/ttf.rs
//!
//! TTF parser producing glyph [`Outline`]s, and the [`Font`] that turns them
//! into coverage `Kernel`s.
//!
//! Parsing is geometry only: a glyph's contours come out as line and
//! quadratic segments in font units, with compound glyphs flattened through
//! their component transforms. Every scale — the em square, the screen flip,
//! a component placement — is applied to control points here, on the host,
//! so that the kernel [`loop_blinn`] builds is in
//! the frame it will be evaluated in. Nothing here is a scene graph of Rust
//! types and nothing here warps a finished kernel.

use super::loop_blinn::{self, Glyph};
use super::outline::{Affine, Contour, Outline};

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

/// Component transform scale factors are 2.14 fixed point.
const F2DOT14: f32 = 16384.0;

/// Glyph flags (simple glyphs): bit 0 on-curve, bit 3 repeat, bits 1/4 and
/// 2/5 the X and Y delta encodings.
const FLAG_ON_CURVE: u8 = 1;
const FLAG_REPEAT: u8 = 8;
const FLAG_X_SHORT: u8 = 2;
const FLAG_X_SAME_OR_POSITIVE: u8 = 16;
const FLAG_Y_SHORT: u8 = 4;
const FLAG_Y_SAME_OR_POSITIVE: u8 = 32;

/// Component flags (compound glyphs).
const COMPONENT_ARGS_ARE_WORDS: u16 = 0x0001;
const COMPONENT_ARGS_ARE_XY_VALUES: u16 = 0x0002;
const COMPONENT_HAVE_A_SCALE: u16 = 0x0008;
const COMPONENT_MORE_COMPONENTS: u16 = 0x0020;
const COMPONENT_HAVE_AN_X_AND_Y_SCALE: u16 = 0x0040;
const COMPONENT_HAVE_A_TWO_BY_TWO: u16 = 0x0080;

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

    /// The glyph for `ch` in font units, as a [`Glyph`]: a coverage
    /// `Kernel` whose winding sum reads a piece table at a
    /// `Kernel::sum_over` binder — the table travels with the kernel itself
    /// (`Kernel::with_buffer_data`), so baking or collapsing it needs no
    /// separate bind; antialiasing resolves from `Dwrt` at bake.
    #[must_use]
    pub fn glyph_kernel(&self, ch: char) -> Option<Glyph> {
        self.glyph_kernel_by_id(self.cmap.lookup(ch as u32)?)
    }

    /// [`Font::glyph_kernel`] by pre-looked-up glyph ID.
    ///
    /// Built in font units, so its antialiasing ramp is one *font unit* wide
    /// and its support is bounded at one font unit past the outline. For a
    /// ramp that is one screen pixel wide, scale the outline before the
    /// kernel exists — [`Font::glyph_scaled_by_id`] — rather than the kernel
    /// after.
    #[must_use]
    pub fn glyph_kernel_by_id(&self, id: u16) -> Option<Glyph> {
        Some(loop_blinn::glyph(&self.outline_by_id(id)?))
    }

    /// The `size`-scaled glyph for `ch` as a [`Glyph`]: the ascent line sits
    /// at screen y=0 (top) and the descent at y=`size`, with screen Y
    /// increasing downward. See [`Font::glyph_kernel`] for the binding this
    /// carries alongside the kernel.
    #[must_use]
    pub fn glyph_kernel_scaled(&self, ch: char, size: f32) -> Option<Glyph> {
        let id = self.cmap.lookup(ch as u32)?;
        self.glyph_kernel_scaled_by_id(id, size)
    }

    /// [`Font::glyph_kernel_scaled`] by pre-looked-up glyph ID.
    #[must_use]
    pub fn glyph_kernel_scaled_by_id(&self, id: u16, size: f32) -> Option<Glyph> {
        self.glyph_scaled_by_id(id, size)
    }

    /// The `size`-scaled glyph **and the box outside which its kernel is
    /// exactly zero**, by pre-looked-up glyph ID.
    ///
    /// The pair is what a caller placing the glyph on a domain-side extent
    /// needs: the range it must give the glyph is the support, and outside it
    /// the glyph may be dropped without changing a bit.
    #[must_use]
    pub fn glyph_scaled_by_id(&self, id: u16, size: f32) -> Option<Glyph> {
        Some(loop_blinn::glyph(&self.outline_scaled_by_id(id, size)?))
    }

    /// The glyph's outline in font units (Y up, as the font stores it).
    ///
    /// `None` when the font has no such glyph; an outline with no contours
    /// for a glyph that draws nothing (a space).
    #[must_use]
    pub fn outline_by_id(&self, id: u16) -> Option<Outline> {
        self.outline(id)
    }

    /// The glyph's outline in the screen frame at `size` pixels: the ascent
    /// line at y=0, the descent line at y=`size`, Y increasing downward.
    #[must_use]
    pub fn outline_scaled_by_id(&self, id: u16, size: f32) -> Option<Outline> {
        Some(self.outline(id)?.transformed(self.to_screen(size)))
    }

    /// The map from font units to the `size`-pixel screen frame: scale by
    /// the total height (ascent + |descent|) so the em fits `size` pixels,
    /// flip Y because screen Y increases downward while font Y increases
    /// upward, and drop the ascent line onto y=0.
    fn to_screen(&self, size: f32) -> Affine {
        let total_height = self.ascent as f32 + self.descent.abs() as f32;
        let scale = size / total_height;
        let ascent_px = self.ascent as f32 * scale;
        Affine([scale, 0.0, 0.0, -scale, 0.0, ascent_px])
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

    /// Parse a glyph's outline in font units. Simple glyphs decode to
    /// contours directly; compound glyphs are the outlines of their
    /// components, each pushed through its component transform. Empty
    /// glyphs are the empty outline.
    fn outline(&self, id: u16) -> Option<Outline> {
        let (a, b) = (self.loca.get(id as usize)?, self.loca.get(id as usize + 1)?);
        if a == b {
            return Some(Outline::default());
        }
        let mut r = R(self.data, self.glyf + a);
        let n = r.i16()?;
        // The header's bounding box is not needed: the outline's own control
        // points bound it, and bound the curves too.
        r.skip(8)?;
        if n >= 0 {
            self.simple(&mut r, n as usize)
        } else {
            self.compound(&mut r)
        }
    }

    /// Decode a simple glyph's point list into contours.
    fn simple(&self, r: &mut R, n: usize) -> Option<Outline> {
        if n == 0 {
            return Some(Outline::default());
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
            if f & FLAG_REPEAT != 0 {
                for _ in 0..r.u8()?.min((np - fl.len()) as u8) {
                    fl.push(f);
                }
            }
        }

        let dec = |r: &mut R, short: u8, same: u8| {
            fl.iter()
                .try_fold((0i16, vec![]), |(mut v, mut out), &f| {
                    v += match (f & short != 0, f & same != 0) {
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

        let xs = dec(r, FLAG_X_SHORT, FLAG_X_SAME_OR_POSITIVE)?;
        let ys = dec(r, FLAG_Y_SHORT, FLAG_Y_SAME_OR_POSITIVE)?;
        let pts: Vec<(f32, f32, bool)> = (0..np)
            .map(|i| (xs[i] as f32, ys[i] as f32, fl[i] & FLAG_ON_CURVE != 0))
            .collect();

        let mut contours = Vec::with_capacity(n);
        let mut start = 0;
        for &e in &ends {
            contours.push(Contour::from_truetype_points(&pts[start..=e])?);
            start = e + 1;
        }
        Some(Outline { contours })
    }

    /// Decode a compound glyph: every component's outline, transformed.
    fn compound(&self, r: &mut R) -> Option<Outline> {
        let mut outline = Outline::default();
        loop {
            let fl = r.u16()?;
            let id = r.u16()?;
            let (dx, dy) = if fl & COMPONENT_ARGS_ARE_XY_VALUES != 0 {
                if fl & COMPONENT_ARGS_ARE_WORDS != 0 {
                    (r.i16()?, r.i16()?)
                } else {
                    (r.i8()? as i16, r.i8()? as i16)
                }
            } else {
                r.skip(if fl & COMPONENT_ARGS_ARE_WORDS != 0 {
                    4
                } else {
                    2
                })?;
                (0, 0)
            };
            let mut m = [1.0, 0.0, 0.0, 1.0, dx as f32, dy as f32];
            if fl & COMPONENT_HAVE_A_SCALE != 0 {
                let s = r.i16()? as f32 / F2DOT14;
                m[0] = s;
                m[3] = s;
            } else if fl & COMPONENT_HAVE_AN_X_AND_Y_SCALE != 0 {
                m[0] = r.i16()? as f32 / F2DOT14;
                m[3] = r.i16()? as f32 / F2DOT14;
            } else if fl & COMPONENT_HAVE_A_TWO_BY_TWO != 0 {
                m[0] = r.i16()? as f32 / F2DOT14;
                m[1] = r.i16()? as f32 / F2DOT14;
                m[2] = r.i16()? as f32 / F2DOT14;
                m[3] = r.i16()? as f32 / F2DOT14;
            }
            if let Some(component) = self.outline(id) {
                outline.append(component.transformed(Affine(m)));
            }
            if fl & COMPONENT_MORE_COMPONENTS == 0 {
                break;
            }
        }
        Some(outline)
    }
}
