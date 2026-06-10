//! Font rendering benchmarks comparing PixelFlow analytical rendering with FreeType.
//!
//! Run with: cargo bench -p pixelflow-graphics

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use pixelflow_graphics::fonts::{text, CachedText, Font, GlyphCache};
use pixelflow_graphics::render::color::{Grayscale, Rgba8};
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::rasterize;

const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

// ============================================================================
// PixelFlow Analytical Rendering Benchmarks
// ============================================================================

fn bench_pixelflow_single_char(c: &mut Criterion) {
    let mut group = c.benchmark_group("pixelflow_single_char");
    let font = Font::parse(FONT_DATA).unwrap();

    // Benchmark different characters to test linear vs quadratic curves
    for (label, ch) in [("A_linear", 'A'), ("O_quadratic", 'O'), ("S_complex", 'S')] {
        group.bench_function(label, |b| {
            let glyph = text(&font, &ch.to_string(), 32.0);
            let colored = Grayscale(glyph);
            let mut frame = Frame::<Rgba8>::new(40, 45);

            b.iter(|| {
                rasterize(black_box(&colored), black_box(&mut frame), 1);
            });
        });
    }

    group.finish();
}

fn bench_pixelflow_text_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("pixelflow_text_sizes");
    let font = Font::parse(FONT_DATA).unwrap();

    for length in [5, 10, 26, 50] {
        let text_str: String = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().take(length).collect();

        group.bench_with_input(BenchmarkId::from_parameter(length), &length, |b, _| {
            let glyph = text(&font, &text_str, 16.0);
            let colored = Grayscale(glyph);
            let width = (length as u32) * 15;
            let mut frame = Frame::<Rgba8>::new(width, 24);

            b.iter(|| {
                rasterize(black_box(&colored), black_box(&mut frame), 1);
            });
        });
    }

    group.finish();
}

fn bench_pixelflow_threading(c: &mut Criterion) {
    let mut group = c.benchmark_group("pixelflow_threading");
    let font = Font::parse(FONT_DATA).unwrap();

    let text_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let glyph = text(&font, text_str, 16.0);
    let colored = Grayscale(glyph);

    for threads in [1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |b, &threads| {
                let mut frame = Frame::<Rgba8>::new(360, 24);

                b.iter(|| {
                    rasterize(black_box(&colored), black_box(&mut frame), threads);
                });
            },
        );
    }

    group.finish();
}

fn bench_pixelflow_with_caching(c: &mut Criterion) {
    let mut group = c.benchmark_group("pixelflow_caching");
    let font = Font::parse(FONT_DATA).unwrap();

    // Compare cached vs uncached text rendering
    group.bench_function("uncached_HELLO", |b| {
        b.iter(|| {
            let glyph = text(&font, "HELLO", 20.0);
            let colored = Grayscale(glyph);
            let mut frame = Frame::<Rgba8>::new(100, 30);
            rasterize(&colored, &mut frame, 1);
        });
    });

    group.bench_function("cached_HELLO", |b| {
        let mut cache = GlyphCache::new();
        let cached = CachedText::new(&font, &mut cache, "HELLO", 20.0);
        let colored = Grayscale(cached);

        b.iter(|| {
            let mut frame = Frame::<Rgba8>::new(100, 30);
            rasterize(black_box(&colored), black_box(&mut frame), 1);
        });
    });

    // Measure cache warm-up overhead
    group.bench_function("cache_warmup_alphabet", |b| {
        b.iter(|| {
            let mut cache = GlyphCache::new();
            for ch in 'A'..='Z' {
                black_box(CachedText::new(&font, &mut cache, &ch.to_string(), 16.0));
            }
        });
    });

    group.finish();
}

// ============================================================================
// FreeType Comparison Benchmarks
// ============================================================================

#[cfg(feature = "freetype")]
fn bench_freetype_single_char(c: &mut Criterion) {
    use freetype as ft;

    let mut group = c.benchmark_group("freetype_single_char");
    let library = ft::Library::init().unwrap();
    let face = library.new_memory_face(FONT_DATA.to_vec(), 0).unwrap();

    for (label, ch) in [("A_linear", 'A'), ("O_quadratic", 'O'), ("S_complex", 'S')] {
        group.bench_function(label, |b| {
            face.set_char_size(0, 32 * 64, 96, 96).unwrap();

            b.iter(|| {
                face.load_char(ch as usize, ft::face::LoadFlag::RENDER)
                    .unwrap();
                let glyph = face.glyph();
                black_box(glyph.bitmap());
            });
        });
    }

    group.finish();
}

#[cfg(feature = "freetype")]
fn bench_freetype_text(c: &mut Criterion) {
    use freetype as ft;

    let mut group = c.benchmark_group("freetype_text");
    let library = ft::Library::init().unwrap();
    let face = library.new_memory_face(FONT_DATA.to_vec(), 0).unwrap();
    face.set_char_size(0, 16 * 64, 96, 96).unwrap();

    for length in [5, 10, 26] {
        let text_str: String = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().take(length).collect();

        group.bench_with_input(BenchmarkId::from_parameter(length), &length, |b, _| {
            b.iter(|| {
                for ch in text_str.chars() {
                    face.load_char(ch as usize, ft::face::LoadFlag::RENDER)
                        .unwrap();
                    let glyph = face.glyph();
                    black_box(glyph.bitmap());
                }
            });
        });
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    pixelflow_benches,
    bench_pixelflow_single_char,
    bench_pixelflow_text_sizes,
    bench_pixelflow_threading,
    bench_pixelflow_with_caching,
);

#[cfg(feature = "freetype")]
criterion_group!(
    freetype_benches,
    bench_freetype_single_char,
    bench_freetype_text,
);

#[cfg(feature = "freetype")]
criterion_main!(pixelflow_benches, freetype_benches);

#[cfg(not(feature = "freetype"))]
criterion_main!(pixelflow_benches);
