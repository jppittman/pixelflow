//! Font rendering benchmarks comparing PixelFlow kernel rendering with FreeType.
//!
//! Run with: cargo bench -p pixelflow-graphics

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use pixelflow_core::{Lattice, Manifold};
use pixelflow_graphics::fonts::{text, text_union, CachedText, Font, GlyphCache};

const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

// ============================================================================
// PixelFlow Kernel Rendering Benchmarks
// ============================================================================

fn bench_pixelflow_single_char(c: &mut Criterion) {
    let mut group = c.benchmark_group("pixelflow_single_char");
    let font = Font::parse(FONT_DATA).unwrap();

    // Different characters exercise linear vs quadratic curve solvers. The
    // glyph is one fused kernel; the JIT compile is cached, so iterations
    // measure the tabulation (the per-frame cost).
    for (label, ch) in [("A_linear", 'A'), ("O_quadratic", 'O'), ("S_complex", 'S')] {
        group.bench_function(label, |b| {
            let kernel = text(&font, &ch.to_string(), 32.0);
            let lattice = Lattice {
                extent: [40, 45, 1, 1],
                origin: [0.5, 0.5, 0.0, 0.0],
            };

            b.iter(|| black_box(lattice.bake(black_box(&kernel))));
        });
    }

    group.finish();
}

/// Long enough that `take(n)` is `n` characters for every length benchmarked.
/// It used to be the 26-letter alphabet, so `text_sizes/50` rendered 26
/// characters over a 50-character frame and its curve was not the flattening
/// it appeared to be.
const SPECIMEN: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\
                        0123456789 The quick brown fox jumps over the lazy dog";

fn bench_pixelflow_text_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("pixelflow_text_sizes");
    let font = Font::parse(FONT_DATA).unwrap();

    for length in [5, 10, 26, 50] {
        let text_str: String = SPECIMEN.chars().take(length).collect();
        assert_eq!(
            text_str.chars().count(),
            length,
            "the specimen is shorter than the length this case claims to render"
        );
        let lattice = Lattice {
            extent: [(length as u32) * 15, 24, 1, 1],
            origin: [0.5, 0.5, 0.0, 0.0],
        };

        // The range encoding: one kernel, every pixel evaluating every glyph.
        group.bench_with_input(BenchmarkId::new("sum", length), &length, |b, _| {
            let kernel = text(&font, &text_str, 16.0);
            b.iter(|| black_box(lattice.bake(black_box(&kernel))));
        });

        // The domain encoding: one program per cell, each collapsed only over
        // the columns its glyphs can reach. `compile` is hoisted for the same
        // reason `cached_HELLO` hoists its own — the measurement is the
        // collapse, and the sum row leaves its compile to the global cache.
        group.bench_with_input(BenchmarkId::new("union", length), &length, |b, _| {
            let program = text_union(&font, lattice, &text_str, 16.0).compile();
            b.iter(|| black_box(program.collapse()));
        });
    }

    group.finish();
}

fn bench_pixelflow_with_caching(c: &mut Criterion) {
    let mut group = c.benchmark_group("pixelflow_caching");
    let font = Font::parse(FONT_DATA).unwrap();

    // Uncached: compose the text kernel and bake it (compile is cached across
    // iterations; construction + tabulation dominate).
    group.bench_function("uncached_HELLO", |b| {
        let lattice = Lattice {
            extent: [100, 30, 1, 1],
            origin: [0.5, 0.5, 0.0, 0.0],
        };
        b.iter(|| {
            let kernel = text(&font, "HELLO", 20.0);
            black_box(lattice.bake(&kernel));
        });
    });

    // Cached: `CachedText` composes already-baked glyph samplers into one
    // kernel over their coverage buffers, so what is timed here is the
    // gather, against the uncached row's curve solve over the same lattice.
    // The compile and the bind are hoisted out of the loop for the same
    // reason the uncached row leaves its compile to the global cache: the
    // measurement is the collapse.
    group.bench_function("cached_HELLO", |b| {
        let mut cache = GlyphCache::new();
        let cached = CachedText::new(&font, &mut cache, "HELLO", 20.0, 1.0);
        let lattice = Lattice {
            extent: [100, 30, 1, 1],
            origin: [0.5, 0.5, 0.0, 0.0],
        };
        let bound = Manifold::compile(&cached.kernel(), lattice.extent).bind(&cached.bindings());

        b.iter(|| black_box(lattice.collapse(black_box(&bound))));
    });

    // Measure cache warm-up overhead (bakes one fused kernel per glyph).
    group.bench_function("cache_warmup_alphabet", |b| {
        b.iter(|| {
            let mut cache = GlyphCache::new();
            for ch in 'A'..='Z' {
                black_box(CachedText::new(
                    &font,
                    &mut cache,
                    &ch.to_string(),
                    16.0,
                    1.0,
                ));
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

    for length in [5, 10, 26, 50] {
        let text_str: String = SPECIMEN.chars().take(length).collect();

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
