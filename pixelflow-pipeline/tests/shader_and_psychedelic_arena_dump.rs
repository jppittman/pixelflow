//! Telemetry dumper, stage 3 (docs/results/2026-09-01-rule-order-real-kernels.md):
//! write the 12 `shader_bench` kernels and the psychedelic shader kernel to
//! the same text arena-dump format `pixelflow-core`'s and
//! `pixelflow-graphics`'s dumpers use, so
//! `pixelflow-search/src/runtime.rs`'s rule-order harness can load them
//! alongside the cell-grid and glyph dumps and replay the exact production
//! saturation sequence on them, under every rule-order arm.
//!
//! The 12 shaders are already `(ExprArena, ExprId)` pairs
//! (`pixelflow_pipeline::shader_bench::named_shadertoy_kernel`) — dumped
//! verbatim, no lowering needed (they were built by hand for the corpus,
//! never through the `kernel!` macro's own e-graph).
//!
//! The psychedelic kernel (`pixelflow-compiler/src/codegen/mod.rs`'s
//! `emit_exact_psychedelic_kernel` test / `pixelflow-runtime/examples/
//! psychedelic_shader.rs`'s `PsychedelicScene`) is hand-transcribed here
//! into the same `ExprArena` `push_*` calls `shader_bench.rs` uses, because
//! `pixelflow-compiler`'s parser/sema modules are private (getting an
//! unoptimized arena out of the `kernel!` pipeline from outside the crate
//! would be the public-API change CLAUDE.md forbids). `t`/`width`/`height`
//! are the kernel's macro parameters, held fixed at representative values
//! (a uniform parameter is a per-compile constant, not a per-pixel
//! coordinate, so this matches how every runtime bake fixes them too); `X`
//! and `Y` are the screen coordinates and `W` stands in for `t`'s only
//! per-pixel-varying use (`time = W + t`, matching `shader_bench.rs`'s own
//! "a coordinate stands in for iTime" convention in `plasma`/
//! `domain_warp_fbm`). The three RGB channels are summed into one scalar
//! root — this corpus's kernels are single-channel, exactly as
//! `shader_bench.rs`'s `cosine_palette` states for the same reason.

use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use pixelflow_pipeline::shader_bench::{SHADERTOY_KERNEL_NAMES, named_shadertoy_kernel};

/// Screen size and time offset the psychedelic kernel is fixed at for this
/// dump — representative values, not load-bearing (the kernel is pure
/// arithmetic in X/Y/W regardless of what these constants are).
const WIDTH: f32 = 800.0;
const HEIGHT: f32 = 600.0;
const T: f32 = 1.0;

fn k(a: &mut ExprArena, v: f32) -> ExprId {
    a.push_const(v)
}
fn var(a: &mut ExprArena, i: u8) -> ExprId {
    a.push_var(i)
}
fn add(a: &mut ExprArena, x: ExprId, y: ExprId) -> ExprId {
    a.push_binary(OpKind::Add, x, y)
}
fn sub(a: &mut ExprArena, x: ExprId, y: ExprId) -> ExprId {
    a.push_binary(OpKind::Sub, x, y)
}
fn mul(a: &mut ExprArena, x: ExprId, y: ExprId) -> ExprId {
    a.push_binary(OpKind::Mul, x, y)
}
fn div(a: &mut ExprArena, x: ExprId, y: ExprId) -> ExprId {
    a.push_binary(OpKind::Div, x, y)
}
fn abs(a: &mut ExprArena, x: ExprId) -> ExprId {
    a.push_unary(OpKind::Abs, x)
}
fn sin(a: &mut ExprArena, x: ExprId) -> ExprId {
    a.push_unary(OpKind::Sin, x)
}
fn exp(a: &mut ExprArena, x: ExprId) -> ExprId {
    a.push_unary(OpKind::Exp, x)
}

/// Exact transcription of `emit_exact_psychedelic_kernel`'s expression
/// (`pixelflow-compiler/src/codegen/mod.rs`), one `let` at a time, into
/// direct `ExprArena` calls. See the module doc for what's fixed vs.
/// per-pixel.
fn psychedelic_kernel() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x_coord = var(&mut a, 0); // X
    let y_coord = var(&mut a, 1); // Y
    let w_coord = var(&mut a, 3); // W

    let height = k(&mut a, HEIGHT);
    let width = k(&mut a, WIDTH);
    let two = k(&mut a, 2.0);
    let scale = div(&mut a, two, height);
    let half = k(&mut a, 0.5);
    let half_width = mul(&mut a, width, half);
    let half_height = mul(&mut a, height, half);
    let x = {
        let d = sub(&mut a, x_coord, half_width);
        mul(&mut a, d, scale)
    };
    let y = {
        let d = sub(&mut a, half_height, y_coord);
        mul(&mut a, d, scale)
    };

    let t = k(&mut a, T);
    let time = add(&mut a, w_coord, t);

    let r_sq = {
        let xx = mul(&mut a, x, x);
        let yy = mul(&mut a, y, y);
        add(&mut a, xx, yy)
    };
    let point7 = k(&mut a, 0.7);
    let radial = {
        let d = sub(&mut a, r_sq, point7);
        abs(&mut a, d)
    };

    let one = k(&mut a, 1.0);
    let five = k(&mut a, 5.0);
    let swirl_scale = {
        let d = sub(&mut a, one, radial);
        mul(&mut a, d, five)
    };
    let vx = mul(&mut a, x, swirl_scale);
    let vy = mul(&mut a, y, swirl_scale);

    let point5 = k(&mut a, 0.5);
    let phase = mul(&mut a, time, point5);
    let point3 = k(&mut a, 0.3);
    let sin_w03 = {
        let arg = mul(&mut a, time, point3);
        sin(&mut a, arg)
    };
    let sin_w20 = {
        let arg = mul(&mut a, time, two);
        sin(&mut a, arg)
    };

    // ((vx + phase).sin() + 1.0) * ((vx + phase) - (vy + phase * 0.7)).abs() * 0.2 + 0.001
    let point2 = k(&mut a, 0.2);
    let point001 = k(&mut a, 0.001);
    let point7b = k(&mut a, 0.7);
    let swirl = {
        let vx_phase = add(&mut a, vx, phase);
        let s = sin(&mut a, vx_phase);
        let s1 = add(&mut a, s, one);
        let phase_07 = mul(&mut a, phase, point7b);
        let vy_phase07 = add(&mut a, vy, phase_07);
        let diff = sub(&mut a, vx_phase, vy_phase07);
        let ad = abs(&mut a, diff);
        let m1 = mul(&mut a, s1, ad);
        let m2 = mul(&mut a, m1, point2);
        add(&mut a, m2, point001)
    };

    let point1 = k(&mut a, 0.1);
    let pulse = {
        let p = mul(&mut a, sin_w20, point1);
        add(&mut a, one, p)
    };
    let neg4 = k(&mut a, -4.0);
    let radial_factor = {
        let r4 = mul(&mut a, radial, neg4);
        let rp = mul(&mut a, r4, pulse);
        exp(&mut a, rp)
    };

    /// The `y_factor_{r,g,b}`/`raw_{r,g,b}`/`soft_{r,g,b}` chain shared by
    /// the three channels — only `y_mult` differs between them.
    struct ChannelInputs {
        y: ExprId,
        sin_w03: ExprId,
        radial_factor: ExprId,
        swirl: ExprId,
    }
    fn channel(a: &mut ExprArena, inputs: &ChannelInputs, y_mult: f32) -> ExprId {
        let ym = k(a, y_mult);
        let yy = mul(a, inputs.y, ym);
        let p2 = k(a, 0.2);
        let sw = mul(a, inputs.sin_w03, p2);
        let sum = add(a, yy, sw);
        let y_factor = exp(a, sum);
        let raw = {
            let m = mul(a, y_factor, inputs.radial_factor);
            div(a, m, inputs.swirl)
        };
        let raw_abs = abs(a, raw);
        let one = k(a, 1.0);
        let denom = add(a, raw_abs, one);
        let soft = div(a, raw, denom);
        let half = k(a, 0.5);
        let s1 = add(a, soft, one);
        mul(a, s1, half)
    }

    let channel_inputs = ChannelInputs {
        y,
        sin_w03,
        radial_factor,
        swirl,
    };
    let red = channel(&mut a, &channel_inputs, 1.0);
    let green = channel(&mut a, &channel_inputs, -1.0);
    let blue = channel(&mut a, &channel_inputs, -2.0);

    let rg = add(&mut a, red, green);
    let root = add(&mut a, rg, blue);
    (a, root)
}

/// Duplicated verbatim from `pixelflow-graphics/tests/production_glyph_arena_dump.rs`
/// / `pixelflow-core/src/lattice/cell_grid.rs` — see either for why (the
/// only crate all three dumpers can see is `pixelflow-ir`, which must not
/// grow a test-only serializer).
fn dump_arena(arena: &ExprArena, root: ExprId, name: &str, path: &std::path::Path) {
    use std::fmt::Write as _;
    let len = arena.nodes_raw().len();
    let mut reachable = vec![false; len];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut reachable[id.0 as usize], true) {
            continue;
        }
        stack.extend(arena.children(id));
    }
    let mut out = String::new();
    writeln!(out, "# pixelflow arena dump v1").expect("fmt");
    writeln!(out, "name {name}").expect("fmt");
    // Every kernel dumped by this file is pure arithmetic — no buffers.
    assert!(
        arena.buffers().is_empty(),
        "{name}: unexpected buffer in a shader/psychedelic kernel"
    );
    let mut dense: Vec<u32> = vec![u32::MAX; len];
    let mut next = 0u32;
    let d = |dense: &[u32], id: ExprId| -> u32 {
        let v = dense[id.0 as usize];
        assert_ne!(v, u32::MAX, "child dumped before parent");
        v
    };
    for idx in 0..len {
        if !reachable[idx] {
            continue;
        }
        let id = ExprId(idx as u32);
        match arena.node(id) {
            ExprNode::Var(i) => writeln!(out, "V {i}"),
            ExprNode::Const(v) => writeln!(out, "C {}", v.to_bits()),
            ExprNode::Buffer(b) => writeln!(out, "B {}", b.0),
            ExprNode::Unary(k, a) => writeln!(out, "U {k:?} {}", d(&dense, *a)),
            ExprNode::Binary(k, a, b) => {
                writeln!(out, "Bi {k:?} {} {}", d(&dense, *a), d(&dense, *b))
            }
            ExprNode::Ternary(k, a, b, c) => writeln!(
                out,
                "T {k:?} {} {} {}",
                d(&dense, *a),
                d(&dense, *b),
                d(&dense, *c)
            ),
            other => panic!("{name}: unsupported node in dump: {other:?}"),
        }
        .expect("fmt");
        dense[idx] = next;
        next += 1;
    }
    writeln!(out, "root {}", d(&dense, root)).expect("fmt");
    std::fs::write(path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn reachable_count(arena: &ExprArena, root: ExprId) -> usize {
    let len = arena.nodes_raw().len();
    let mut seen = vec![false; len];
    let mut stack = vec![root];
    let mut n = 0;
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        n += 1;
        stack.extend(arena.children(id));
    }
    n
}

#[test]
#[ignore = "telemetry dumper: PIXELFLOW_TELEMETRY_DIR=<dir> cargo test -p pixelflow-pipeline --release --test shader_and_psychedelic_arena_dump -- --ignored"]
fn dump_shader_and_psychedelic_arenas() {
    let dir = std::path::PathBuf::from(
        std::env::var("PIXELFLOW_TELEMETRY_DIR").expect("PIXELFLOW_TELEMETRY_DIR must be set"),
    );
    std::fs::create_dir_all(&dir).expect("create dump dir");

    let mut dumped = 0usize;
    for name in SHADERTOY_KERNEL_NAMES {
        let (arena, root) = named_shadertoy_kernel(name)
            .unwrap_or_else(|| panic!("{name}: not found in shader_bench"));
        let dump_name = format!("shader:{name}");
        let path = dir.join(format!("shader_{name}.arena"));
        println!(
            "{dump_name}: {} reachable nodes -> {}",
            reachable_count(&arena, root),
            path.display()
        );
        dump_arena(&arena, root, &dump_name, &path);
        dumped += 1;
    }

    let (arena, root) = psychedelic_kernel();
    let path = dir.join("psychedelic.arena");
    println!(
        "psychedelic: {} reachable nodes -> {}",
        reachable_count(&arena, root),
        path.display()
    );
    dump_arena(&arena, root, "psychedelic", &path);
    dumped += 1;

    assert_eq!(dumped, SHADERTOY_KERNEL_NAMES.len() + 1);
    println!(
        "dumped {dumped} shader/psychedelic arenas to {}",
        dir.display()
    );
}
