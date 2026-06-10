//! Benchmarks for the ANSI parser.
//!
//! Run with: cargo bench -p core-term

use core_term::ansi::{AnsiParser, AnsiProcessor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Pure ASCII text - fast path
fn ascii_text(size: usize) -> Vec<u8> {
    "The quick brown fox jumps over the lazy dog. "
        .bytes()
        .cycle()
        .take(size)
        .collect()
}

/// Heavy CSI sequences (cursor movement, SGR)
fn csi_heavy(size: usize) -> Vec<u8> {
    // ESC[1;31m (red) + "X" + ESC[0m (reset) = 12 bytes per cycle
    let seq = b"\x1b[1;31mX\x1b[0m";
    seq.iter().copied().cycle().take(size).collect()
}

/// SGR color cycling (256 colors)
fn sgr_256_colors(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    let mut i = 0u8;
    while data.len() < size {
        // ESC[38;5;Nm where N cycles 0-255
        let seq = format!("\x1b[38;5;{}m.", i);
        data.extend_from_slice(seq.as_bytes());
        i = i.wrapping_add(1);
    }
    data.truncate(size);
    data
}

/// Cursor movement storm
fn cursor_movement(size: usize) -> Vec<u8> {
    // ESC[H (home) + ESC[5;10H (goto) + ESC[A (up) + ESC[B (down)
    let seq = b"\x1b[H\x1b[5;10H\x1b[A\x1b[B";
    seq.iter().copied().cycle().take(size).collect()
}

/// vtebench-style: alt screen random write
fn vtebench_alt_screen(size: usize) -> Vec<u8> {
    // ESC[?1049h (alt screen) + random positions + chars + ESC[?1049l (exit)
    let mut data = Vec::with_capacity(size);
    data.extend_from_slice(b"\x1b[?1049h"); // Enter alt screen

    let body = b"\x1b[10;20HA\x1b[5;15HB\x1b[1;1HC";
    while data.len() < size - 10 {
        data.extend_from_slice(body);
    }
    data.extend_from_slice(b"\x1b[?1049l"); // Exit alt screen
    data.truncate(size);
    data
}

/// UTF-8 heavy (emoji, CJK)
fn unicode_heavy(size: usize) -> Vec<u8> {
    "こんにちは世界🎉🚀💻".bytes().cycle().take(size).collect()
}

/// OSC sequences (window title, etc)
fn osc_sequences(size: usize) -> Vec<u8> {
    // ESC]0;title\x07 (set window title)
    let seq = b"\x1b]0;Window Title Here\x07";
    seq.iter().copied().cycle().take(size).collect()
}

/// Scrolling simulation (newlines + text)
fn scrolling(size: usize) -> Vec<u8> {
    "Line of text content here\n"
        .bytes()
        .cycle()
        .take(size)
        .collect()
}

fn bench_parser_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("ansi_parser");
    // Reduce sample count for faster execution
    group.sample_size(50);

    // Test different sizes (reduced from [1024, 4096, 16384, 65536, 262144])
    for size in [1024, 4096, 16384, 65536] {
        // ASCII text (fast path)
        let data = ascii_text(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_with_input(BenchmarkId::new("ascii_text", size), &data, |b, data| {
            b.iter(|| {
                let mut parser = AnsiProcessor::new();
                black_box(parser.process_bytes(data))
            })
        });

        // CSI heavy
        let data = csi_heavy(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_with_input(BenchmarkId::new("csi_heavy", size), &data, |b, data| {
            b.iter(|| {
                let mut parser = AnsiProcessor::new();
                black_box(parser.process_bytes(data))
            })
        });

        // SGR 256 colors
        let data = sgr_256_colors(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("sgr_256_colors", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut parser = AnsiProcessor::new();
                    black_box(parser.process_bytes(data))
                })
            },
        );

        // Cursor movement
        let data = cursor_movement(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("cursor_movement", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut parser = AnsiProcessor::new();
                    black_box(parser.process_bytes(data))
                })
            },
        );

        // Unicode
        let data = unicode_heavy(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_with_input(BenchmarkId::new("unicode_heavy", size), &data, |b, data| {
            b.iter(|| {
                let mut parser = AnsiProcessor::new();
                black_box(parser.process_bytes(data))
            })
        });

        // Scrolling
        let data = scrolling(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_with_input(BenchmarkId::new("scrolling", size), &data, |b, data| {
            b.iter(|| {
                let mut parser = AnsiProcessor::new();
                black_box(parser.process_bytes(data))
            })
        });
    }

    group.finish();
}

fn bench_vtebench_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("vtebench_scenarios");

    let size = 65536; // 64KB chunks like vtebench

    // Alt screen random write
    let data = vtebench_alt_screen(size);
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("alt_screen_random_write", |b| {
        b.iter(|| {
            let mut parser = AnsiProcessor::new();
            black_box(parser.process_bytes(&data))
        })
    });

    // Scrolling
    let data = scrolling(size);
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("scrolling", |b| {
        b.iter(|| {
            let mut parser = AnsiProcessor::new();
            black_box(parser.process_bytes(&data))
        })
    });

    // Unicode random write
    let data = unicode_heavy(size);
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("unicode_random_write", |b| {
        b.iter(|| {
            let mut parser = AnsiProcessor::new();
            black_box(parser.process_bytes(&data))
        })
    });

    // OSC heavy (hyperlinks, titles)
    let data = osc_sequences(size);
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("osc_heavy", |b| {
        b.iter(|| {
            let mut parser = AnsiProcessor::new();
            black_box(parser.process_bytes(&data))
        })
    });

    group.finish();
}

criterion_group!(benches, bench_parser_throughput, bench_vtebench_scenarios);
criterion_main!(benches);
