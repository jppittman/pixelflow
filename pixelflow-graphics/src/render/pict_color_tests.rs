// src/render/pict_color_tests.rs

//! Worked example for the PICT-style generator in [`super::pict`], applied to
//! the renderer's color/pixel layer.
//!
//! Two models are exercised:
//!
//! 1. **Packing algebra** — treat the four channels `(R, G, B, A)` as factors
//!    and check the byte-level pixel operations (`Rgba8`/`Bgra8` construction,
//!    accessors, channel-order conversion, the `Pixel` trait) against algebraic
//!    oracles: round-trips and channel-preservation identities.
//!
//! 2. **Render-pipeline cross-check** — treat `(R, G, B)` as factors and, for
//!    each generated color, compare the two independent implementations of
//!    "what pixel is this color": the direct `u32::from(Color)` pack and the
//!    full pull-based manifold render through the SIMD backend
//!    (`materialize_discrete`). They must agree bit-for-bit.
//!
//! Color packing is a natural fit for combinatorial testing: channel order and
//! endianness bugs are exactly the kind of 2-way interaction PICT targets, and
//! the number of `(R, G, B, A)` combinations (256^4) is far too large to
//! enumerate.

use super::color::{Bgra8, Color, Rgba8};
use super::pict::pairwise;
use super::pixel::Pixel;
use pixelflow_core::{materialize_discrete, PARALLELISM};

/// Representative channel values: the byte-range boundaries plus the midpoint
/// split (127/128) where rounding and sign handling tend to break.
const LEVELS: [u8; 8] = [0, 1, 64, 127, 128, 191, 254, 255];

/// Render a color through the full pull-based manifold pipeline and return the
/// first lane's packed pixel (RGBA byte order: `0xAABBGGRR`).
fn render_color(color: Color) -> u32 {
    let mut out = vec![0u32; PARALLELISM];
    materialize_discrete(&color, 0.0, 0.0, &mut out);
    out[0]
}

fn channels(packed: u32) -> (u8, u8, u8, u8) {
    (
        (packed & 0xFF) as u8,
        ((packed >> 8) & 0xFF) as u8,
        ((packed >> 16) & 0xFF) as u8,
        ((packed >> 24) & 0xFF) as u8,
    )
}

/// Model 1: the byte-packing algebra over four channel factors.
#[test]
fn pict_pixel_packing_algebra() {
    let rows = pairwise(&[LEVELS.len(); 4]);
    let mut failures: Vec<String> = Vec::new();

    let mut check = |cond: bool, msg: String| {
        if !cond {
            failures.push(msg);
        }
    };

    for row in &rows {
        let (r, g, b, a) = (LEVELS[row[0]], LEVELS[row[1]], LEVELS[row[2]], LEVELS[row[3]]);

        // Rgba8 construction and accessors round-trip.
        let rgba = Rgba8::new(r, g, b, a);
        check(
            (rgba.r(), rgba.g(), rgba.b(), rgba.a()) == (r, g, b, a),
            format!("Rgba8::new accessors: ({r},{g},{b},{a}) -> {:?}", (rgba.r(), rgba.g(), rgba.b(), rgba.a())),
        );

        // Bgra8 stores the same logical channels (note the (b,g,r,a) arg order).
        let bgra = Bgra8::new(b, g, r, a);
        check(
            (bgra.r(), bgra.g(), bgra.b(), bgra.a()) == (r, g, b, a),
            format!("Bgra8::new accessors: logical ({r},{g},{b},{a}) -> {:?}", (bgra.r(), bgra.g(), bgra.b(), bgra.a())),
        );

        // Rgba8 <-> Bgra8 preserves logical channels and is involutive.
        let to_bgra = Bgra8::from(rgba);
        check(
            (to_bgra.r(), to_bgra.g(), to_bgra.b(), to_bgra.a()) == (r, g, b, a),
            format!("Rgba8->Bgra8 channel preservation for ({r},{g},{b},{a})"),
        );
        check(
            Rgba8::from(to_bgra) == rgba,
            format!("Rgba8->Bgra8->Rgba8 not involutive for ({r},{g},{b},{a})"),
        );

        // Pixel::from_u32/to_u32 round-trip.
        check(
            Rgba8::from_u32(rgba.to_u32()) == rgba,
            format!("Rgba8 u32 round-trip for ({r},{g},{b},{a})"),
        );

        // Pixel::from_rgba matches direct construction for both formats.
        let norm = |v: u8| v as f32 / 255.0;
        check(
            <Rgba8 as Pixel>::from_rgba(norm(r), norm(g), norm(b), norm(a)) == rgba,
            format!("Rgba8::from_rgba mismatch for ({r},{g},{b},{a})"),
        );
        check(
            <Bgra8 as Pixel>::from_rgba(norm(r), norm(g), norm(b), norm(a)) == bgra,
            format!("Bgra8::from_rgba mismatch for ({r},{g},{b},{a})"),
        );
    }

    assert!(
        failures.is_empty(),
        "PICT pairwise pixel-packing testing found {} failing invariants over {} cases:\n  {}",
        failures.len(),
        rows.len(),
        failures.join("\n  "),
    );
}

/// Model 2: the direct pack and the manifold render must agree, for every
/// pairwise `(R, G, B)` combination.
#[test]
fn pict_render_pipeline_matches_direct_pack() {
    let rows = pairwise(&[LEVELS.len(); 3]);
    let mut failures: Vec<String> = Vec::new();

    for row in &rows {
        let (r, g, b) = (LEVELS[row[0]], LEVELS[row[1]], LEVELS[row[2]]);
        let color = Color::Rgb(r, g, b);

        let direct = u32::from(color);
        let rendered = render_color(color);

        if direct != rendered {
            failures.push(format!(
                "Rgb({r},{g},{b}): direct pack {:?} != rendered {:?}",
                channels(direct),
                channels(rendered),
            ));
            continue;
        }
        // And the rendered pixel must actually carry the requested channels.
        if channels(rendered) != (r, g, b, 255) {
            failures.push(format!(
                "Rgb({r},{g},{b}): rendered channels {:?} != expected ({r},{g},{b},255)",
                channels(rendered),
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "PICT pairwise render cross-check found {} mismatches over {} cases:\n  {}",
        failures.len(),
        rows.len(),
        failures.join("\n  "),
    );
}

/// Exhaustive 1-way sweep of the 256-color palette: the direct pack and the
/// manifold render must agree for every indexed color, too.
#[test]
fn palette_render_matches_direct_pack() {
    let mut failures: Vec<String> = Vec::new();
    for idx in 0u16..=255 {
        let color = Color::Indexed(idx as u8);
        let direct = u32::from(color);
        let rendered = render_color(color);
        if direct != rendered {
            failures.push(format!(
                "Indexed({idx}): direct {:?} != rendered {:?}",
                channels(direct),
                channels(rendered),
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "indexed-color render cross-check found {} mismatches:\n  {}",
        failures.len(),
        failures.join("\n  "),
    );
}
