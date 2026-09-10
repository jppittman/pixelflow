//! Collapse-loop equivalence: the kernel produced by `compile` must
//! agree, bit for bit, with the IR interpreter over the lattice it fills.
//!
//! The oracle is `eval_scalar`, an independent implementation. It used to be a
//! second compiled kernel — the per-batch entry — but both framings wrapped the
//! same `emit_dag_body` output, so that comparison could only ever see the loop
//! scaffold (coordinate slots, X induction, output stores, branch patching) and
//! was blind to any bug in the shared body. The interpreter sees both.
//!
//! Runs at whichever width the build selects (SSE2/AVX2/AVX-512/NEON): the
//! CI ISA matrix covers the x86 widths, macOS covers aarch64.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::emit::{CompileResult, compile};
use pixelflow_codegen::{JIT_VECTOR_BYTES, Point4, TileSlice};
use pixelflow_ir::OpKind;
use pixelflow_ir::Term;
use pixelflow_ir::binding::BindingTable;
use pixelflow_ir::decl::{BufferDecl, BufferIdentity};
use pixelflow_ir::eval_scalar;
use pixelflow_ir::expr::{ExprBuilder, ExprRef};
use pixelflow_ir::{DifferentialCheck, PointVerdict, Uniform};

/// Declare `u` in `a` and return its leaf — a scalar invariant across the
/// lattice, which is what a third and fourth coordinate always were.
fn arg_leaf(a: &mut ExprBuilder, u: Uniform) -> ExprRef {
    let slot = a.declare_uniform(u.decl());
    a.push_uniform(slot)
}

const LANES: usize = JIT_VECTOR_BYTES / 4;

/// One collapse call: fill `out` (whose length must be `groups * LANES`)
/// from lane-sequential X starting at `origin.x`, with loop-invariant y/z/w.
fn run_collapse_grid(
    res: &CompileResult,
    ctx: &[*const f32],
    tile: TileSlice,
    origin: Point4<f32>,
) {
    let mut x0_vec = [0.0f32; LANES];
    for (i, lane) in x0_vec.iter_mut().enumerate() {
        *lane = origin.x + i as f32;
    }
    unsafe {
        res.code.call_collapse(
            ctx.as_ptr(),
            tile,
            Point4::new(
                x0_vec,
                [origin.y; LANES],
                [origin.z; LANES],
                [origin.w; LANES],
            ),
        );
    }
}

fn run_collapse(res: &CompileResult, ctx: &[*const f32], out: &mut [f32], origin: Point4<f32>) {
    assert_eq!(out.len() % LANES, 0);
    let groups = out.len() / LANES;
    run_collapse_grid(
        res,
        ctx,
        TileSlice::contiguous(out.as_mut_ptr(), groups, 1),
        origin,
    );
}

/// Compile the collapse kernel and require it to agree, bit for bit, with the
/// IR interpreter over several rows; returns the result for extra assertions.
///
/// The oracle used to be a second *compiled* kernel (the per-batch entry), but
/// both framings wrap the same `emit_dag_body` output, so that comparison was
/// blind to any bug in the shared body — it could only ever see the scaffold.
/// `eval_scalar` is an independent implementation, so it sees both.
fn assert_collapse_matches_interpreter(
    t: Term<'_>,
    bufs: &[&[f32]],
    groups: usize,
    label: &str,
) -> CompileResult {
    let collapse = compile(t).unwrap_or_else(|e| panic!("{label}: collapse compile failed: {e}"));

    let ctx: Vec<*const f32> = bufs.iter().map(|b| b.as_ptr()).collect();
    let bindings = if bufs.is_empty() {
        BindingTable::empty()
    } else {
        BindingTable::bind(t.env(), bufs).expect("bind test buffers")
    };

    for &(x0, y, z, w) in &[
        (0.5f32, 0.5f32, 0.0f32, 0.0f32),
        (-7.25, 3.5, 0.1, 0.9),
        (100.0, -2.0, 1.0, -1.0),
    ] {
        let mut got = vec![0.0f32; groups * LANES];
        run_collapse(&collapse, &ctx, &mut got, Point4::new(x0, y, z, w));
        for (i, &g) in got.iter().enumerate() {
            let xi = x0 + i as f32;
            let vars = [xi, y];
            let want = eval_scalar(t, &vars, &bindings);
            if (want.is_nan() && g.is_nan()) || g == want {
                continue;
            }
            // Not bit-identical. That is a miscompile for everything this
            // suite builds EXCEPT where the op is one CLAUDE.md's "Floating
            // point at the edges" table already marks target-divergent —
            // `MulAdd` is one rounding where an FMA instruction exists and two
            // where it does not, and the transcendental expansions are Horner
            // chains of exactly that node. The oracle rounds twice, so an
            // FMA-capable build legitimately lands within an ulp instead of on
            // top. `DifferentialCheck` composes the per-node bound that says
            // how far is legal here; a mask root has no such band, so it stays
            // bit-exact.
            assert!(
                !pixelflow_ir::is_mask_valued(t.root()),
                "{label}: mask-valued root differs — collapse {g:?} != interp {want:?} \
                 at lane {i} of ({x0}, {y}, {z}, {w}); masks are bit patterns, never near-misses"
            );
            let point = DifferentialCheck::new(t).at(&vars, &bindings);
            assert_eq!(
                point.verdict(g),
                PointVerdict::Accept,
                "{label}: collapse {g} outside interp {want} ± {bound} at lane {i} of \
                 ({x0}, {y}, {z}, {w})",
                bound = point.error_bound(),
            );
        }
    }
    collapse
}

#[test]
fn arithmetic_row_matches_the_interpreter() {
    // x*y + (x - 1.5): induction X must advance exactly like the per-batch
    // loop's, and Y must survive every iteration in its slot.
    let mut a = ExprBuilder::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let c = a.push_const(1.5);
    let xy = a.push_binary(OpKind::Mul, x, y);
    let xc = a.push_binary(OpKind::Sub, x, c);
    let root = a.push_binary(OpKind::Add, xy, xc);
    let built = a.finish(&[root]);
    let t = Term::new(built.0.entry(), &built.1);

    let res = assert_collapse_matches_interpreter(t, &[], 5, "arith");

    // And anchor to the reference semantics.
    let mut out = vec![0.0f32; 5 * LANES];
    run_collapse(&res, &[], &mut out, Point4::new(2.0, 3.0, 0.0, 0.0));
    for (i, &got) in out.iter().enumerate() {
        let xi = 2.0 + i as f32;
        let want = eval_scalar(t, &[xi, 3.0], &BindingTable::empty());
        assert_eq!(got, want, "arith vs interp at lane {i}");
    }
}

#[test]
fn two_dimensional_collapse_advances_y_preserves_stride_and_hoists_both_scopes() {
    // sqrt(u*v + 2), over the kernel's two arguments, is invariant across
    // the whole X/Y nest; sqrt(y + 9) is invariant only across each X row.
    // Both feed the X-dependent body.
    let mut a = ExprBuilder::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let (u, v) = (Uniform::new(0.0), Uniform::new(0.0));
    let (z, w) = (arg_leaf(&mut a, u), arg_leaf(&mut a, v));
    let two = a.push_const(2.0);
    let nine = a.push_const(9.0);
    let zw = a.push_binary(OpKind::Mul, z, w);
    let zw2 = a.push_binary(OpKind::Add, zw, two);
    let frame_value = a.push_unary(OpKind::Sqrt, zw2);
    let y9 = a.push_binary(OpKind::Add, y, nine);
    let row_value = a.push_unary(OpKind::Sqrt, y9);
    let fx = a.push_binary(OpKind::Mul, frame_value, x);
    let root = a.push_binary(OpKind::Add, fx, row_value);
    let built = a.finish(&[root]);
    let t = Term::new(built.0.entry(), &built.1);

    let compiled = compile(t).expect("2D collapse compile");
    assert!(
        compiled.hoisted_values >= 2,
        "frame and row roots must both hoist, got {}",
        compiled.hoisted_values
    );

    const GROUPS: usize = 3;
    const ROWS: usize = 4;
    const TAIL: usize = 2;
    let row_len = GROUPS * LANES + TAIL;
    let mut out = vec![-1234.5f32; ROWS * row_len];
    let (x0, y0) = (0.25f32, 1.5f32);
    let block = [2.0f32, 3.0f32];
    let ctx: [*const f32; 1] = [block.as_ptr()];
    let bindings = BindingTable::empty()
        .bind_uniforms(
            &built.1,
            &[(u.identity(), block[0]), (v.identity(), block[1])],
        )
        .expect("both arguments are declared");
    run_collapse_grid(
        &compiled,
        &ctx,
        TileSlice::new(
            out.as_mut_ptr(),
            GROUPS,
            ROWS,
            TAIL * core::mem::size_of::<f32>(),
        ),
        Point4::new(x0, y0, 0.0, 0.0),
    );

    for row in 0..ROWS {
        for col in 0..GROUPS * LANES {
            let got = out[row * row_len + col];
            let want = eval_scalar(t, &[x0 + col as f32, y0 + row as f32], &bindings);
            assert_eq!(got, want, "2D collapse at ({col},{row})");
        }
        assert!(
            out[row * row_len + GROUPS * LANES..(row + 1) * row_len]
                .iter()
                .all(|&v| v == -1234.5),
            "row {row} tail padding was overwritten"
        );
    }
}

#[test]
fn select_short_circuit_branches_inside_loop() {
    // select(x < t, x*2, y): different batches take different guard paths
    // (all-true, mixed, all-false), so the short-circuit branches and the
    // loop's own branches must patch independently.
    let mut a = ExprBuilder::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let t = a.push_const(8.0);
    let cond = a.push_binary(OpKind::Lt, x, t);
    let two = a.push_const(2.0);
    let x2 = a.push_binary(OpKind::Mul, x, two);
    let root = a.push_ternary(OpKind::Select, cond, x2, y);
    let built = a.finish(&[root]);
    let t = Term::new(built.0.entry(), &built.1);

    assert_collapse_matches_interpreter(t, &[], 6, "select");
}

#[test]
fn transcendentals_rematerialize_per_iteration() {
    // sin/exp expand to polynomial chains full of constants: on aarch64 they
    // load X17-relative from the pool anchored in the collapse prologue, on
    // x86 they embed/broadcast — every iteration, inside the loop.
    let mut a = ExprBuilder::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let s = a.push_const(0.05);
    let xs = a.push_binary(OpKind::Mul, x, s);
    let sin = a.push_unary(OpKind::Sin, xs);
    let ey = a.push_unary(OpKind::Exp, y);
    let root = a.push_binary(OpKind::Add, sin, ey);
    let built = a.finish(&[root]);
    let t = Term::new(built.0.entry(), &built.1);

    assert_collapse_matches_interpreter(t, &[], 4, "transcendental");
}

#[test]
fn gather_reads_ctx_every_iteration() {
    // The context register must survive the whole loop: a buffer read per
    // batch whose indices depend on the induction X.
    let w = 64usize;
    let buf: Vec<f32> = (0..w * 2).map(|k| (k as f32) * 0.25 - 3.0).collect();

    let mut a = ExprBuilder::new();
    let b = a.declare_buffer(BufferDecl {
        id: BufferIdentity::mint(),
        width: w as u32,
        height: 2,
    });
    let x = a.push_var(0);
    let y = a.push_var(1);
    let g = a.push_gather(b, x, y);
    let x01 = a.push_const(0.125);
    let xf = a.push_binary(OpKind::Mul, x, x01);
    let root = a.push_binary(OpKind::Add, g, xf);
    let built = a.finish(&[root]);
    let t = Term::new(built.0.entry(), &built.1);

    let ctx = [buf.as_ptr()];
    let res = assert_collapse_matches_interpreter(t, &[buf.as_slice()], 4, "gather");

    // Interpreter anchor over one row.
    let bindings = BindingTable::bind(&built.1, &[buf.as_slice()]).unwrap();
    let mut out = vec![0.0f32; 4 * LANES];
    run_collapse(&res, &ctx, &mut out, Point4::new(0.0, 1.0, 0.0, 0.0));
    for (i, &got) in out.iter().enumerate() {
        let want = eval_scalar(t, &[i as f32, 1.0], &bindings);
        assert_eq!(got, want, "gather vs interp at lane {i}");
    }
}

#[test]
fn matmul_reduce_one_call_fills_output() {
    // out(j) = Σ_i W(i,j) * input(i): the reduce unrolls to a gather/mul/add
    // chain, and ONE collapse call fills the whole output vector.
    let (in_dim, out_dim) = (5usize, 4 * LANES);
    let w: Vec<f32> = (0..(in_dim * out_dim)).map(|k| (k as f32).sin()).collect();
    let input: Vec<f32> = (0..in_dim).map(|k| (k as f32 - 2.0) * 0.5).collect();

    let mut a = ExprBuilder::new();
    let wb = a.declare_buffer(BufferDecl {
        id: BufferIdentity::mint(),
        width: in_dim as u32,
        height: out_dim as u32,
    });
    let ib = a.declare_buffer(BufferDecl {
        id: BufferIdentity::mint(),
        width: in_dim as u32,
        height: 1,
    });
    let i = a.push_var(4);
    let j = a.push_var(0);
    let zero = a.push_const(0.0);
    let wg = a.push_gather(wb, i, j);
    let ig = a.push_gather(ib, i, zero);
    let prod = a.push_binary(OpKind::Mul, wg, ig);
    let root = a.push_reduce(OpKind::Add, 4, in_dim as u32, prod);
    let built = a.finish(&[root]);
    let t = Term::new(built.0.entry(), &built.1);

    let ctx = [w.as_ptr(), input.as_ptr()];
    let res = compile(t).expect("matmul collapse compile");

    let mut out = vec![0.0f32; out_dim];
    run_collapse(&res, &ctx, &mut out, Point4::new(0.0, 0.0, 0.0, 0.0));

    let bindings = BindingTable::bind(&built.1, &[w.as_slice(), input.as_slice()]).unwrap();
    for (j, &got) in out.iter().enumerate() {
        let want = eval_scalar(t, &[j as f32, 0.0], &bindings);
        assert_eq!(got, want, "matmul out[{j}]");
    }
}

#[test]
fn hoisted_row_constants_match_the_interpreter() {
    // sqrt(y*y + 2) * x + y/3: the sqrt chain and the division are X-invariant
    // and consumed by X-dependent ops, so LICM parks them in hoist slots and
    // the loop reloads them. Hoisting reorders nothing within a value's own
    // computation, so the result stays bit-exact against the per-batch kernel
    // (which recomputes them every batch).
    let mut a = ExprBuilder::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let two = a.push_const(2.0);
    let yy = a.push_binary(OpKind::Mul, y, y);
    let yy2 = a.push_binary(OpKind::Add, yy, two);
    let s = a.push_unary(OpKind::Sqrt, yy2);
    let three = a.push_const(3.0);
    let yd = a.push_binary(OpKind::Div, y, three);
    let sx = a.push_binary(OpKind::Mul, s, x);
    let root = a.push_binary(OpKind::Add, sx, yd);
    let built = a.finish(&[root]);
    let t = Term::new(built.0.entry(), &built.1);

    let res = assert_collapse_matches_interpreter(t, &[], 5, "hoist");
    assert!(
        res.hoisted_values >= 2,
        "sqrt chain and division must hoist, got {}",
        res.hoisted_values
    );
}

#[test]
fn fully_invariant_kernel_degenerates_to_a_store_loop() {
    // sqrt(y + u*v): no X anywhere — the whole kernel hoists and the loop
    // body is a bare reload-and-store of the one parked value. `u` and `v`
    // are the kernel's arguments, which are invariant across the lattice
    // outright.
    let mut a = ExprBuilder::new();
    let y = a.push_var(1);
    let (u, v) = (Uniform::new(0.0), Uniform::new(0.0));
    let (z, w) = (arg_leaf(&mut a, u), arg_leaf(&mut a, v));
    let zw = a.push_binary(OpKind::Mul, z, w);
    let yzw = a.push_binary(OpKind::Add, y, zw);
    let root = a.push_unary(OpKind::Sqrt, yzw);
    let built = a.finish(&[root]);
    let t = Term::new(built.0.entry(), &built.1);

    let compiled = compile(t).expect("invariant-root compile");
    assert!(compiled.hoisted_values >= 1);

    let block = [0.1f32, 0.9f32];
    let ctx: [*const f32; 1] = [block.as_ptr()];
    let bindings = BindingTable::empty()
        .bind_uniforms(
            &built.1,
            &[(u.identity(), block[0]), (v.identity(), block[1])],
        )
        .expect("both arguments are declared");
    for &(x0, yv) in &[(0.5f32, 0.5f32), (-7.25, 3.5), (100.0, -2.0)] {
        let mut got = vec![0.0f32; 4 * LANES];
        run_collapse(&compiled, &ctx, &mut got, Point4::new(x0, yv, 0.0, 0.0));
        for (i, &g) in got.iter().enumerate() {
            let want = eval_scalar(t, &[x0 + i as f32, yv], &bindings);
            assert_eq!(g.to_bits(), want.to_bits(), "invariant-root at lane {i}");
        }
    }
}

#[test]
fn hoisted_guarded_select_still_fills_its_slot() {
    // select(y < 4, sqrt(y+9), y*2) * x + select(x < 4, x, y): the first
    // select is X-invariant — it hoists, and the prologue must emit it
    // UNGUARDED: a uniform-mask short-circuit in the prologue would skip an
    // arm's defs (and on the root-hoisted path, the parking store itself),
    // leaving the slot garbage for every loop iteration. The second select
    // stays in the loop with its guards, proving guard machinery still works
    // beside hoisting. Rows are chosen so the invariant select's mask is
    // uniform-true, uniform-false, and (per-batch) mixed across the suite's
    // three (x0, y) rows.
    let mut a = ExprBuilder::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let four = a.push_const(4.0);
    let nine = a.push_const(9.0);
    let two = a.push_const(2.0);

    let ymask = a.push_binary(OpKind::Lt, y, four);
    let y9 = a.push_binary(OpKind::Add, y, nine);
    let sq = a.push_unary(OpKind::Sqrt, y9);
    let y2 = a.push_binary(OpKind::Mul, y, two);
    let ysel = a.push_ternary(OpKind::Select, ymask, sq, y2);

    let xmask = a.push_binary(OpKind::Lt, x, four);
    let xsel = a.push_ternary(OpKind::Select, xmask, x, y);

    let prod = a.push_binary(OpKind::Mul, ysel, x);
    let root = a.push_binary(OpKind::Add, prod, xsel);
    let built = a.finish(&[root]);
    let t = Term::new(built.0.entry(), &built.1);

    let res = assert_collapse_matches_interpreter(t, &[], 6, "hoisted-select");
    assert!(
        res.hoisted_values >= 1,
        "the invariant select must hoist, got {}",
        res.hoisted_values
    );
}

#[test]
fn deep_invariant_chain_spills_in_the_prologue() {
    // A Y-only chain long enough to overflow the register budget: the
    // PROLOGUE's own spill frame must coexist with the loop body's frame
    // (they share the region below the coordinate slots) and with the hoist
    // slots above. The X side is kept wide too so both frames are real.
    let mut a = ExprBuilder::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let mut y_terms = Vec::new();
    let mut x_terms = Vec::new();
    for k in 0..10 {
        let c = a.push_const(0.3 + k as f32);
        let yk = a.push_binary(OpKind::Add, y, c);
        let ys = a.push_unary(OpKind::Sqrt, yk);
        y_terms.push(ys);
        let xk = a.push_binary(OpKind::Mul, x, c);
        x_terms.push(xk);
    }
    // Pair each hoisted sqrt with an X term so every one crosses the loop
    // boundary, then sum with a balanced tree to keep many values live.
    let mut sums: Vec<ExprRef> = y_terms
        .iter()
        .zip(&x_terms)
        .map(|(ys, xk)| a.push_binary(OpKind::Mul, *ys, *xk))
        .collect();
    while sums.len() > 1 {
        let mut next = Vec::new();
        for pair in sums.chunks(2) {
            next.push(match pair {
                [l, r] => a.push_binary(OpKind::Add, *l, *r),
                [l] => *l,
                _ => unreachable!(),
            });
        }
        sums = next;
    }
    let root = sums[0];
    let built = a.finish(&[root]);
    let t = Term::new(built.0.entry(), &built.1);

    let res = assert_collapse_matches_interpreter(t, &[], 3, "deep-hoist");
    assert!(
        res.hoisted_values >= 10,
        "all ten sqrt chains must hoist, got {}",
        res.hoisted_values
    );
}

#[test]
fn spill_frame_coexists_with_coordinate_slots() {
    // Force spilling (more simultaneously-live values than the allocator's
    // budget) so the body's spill frame and the scaffold's coordinate slots
    // must share the stack without aliasing — in both x86 frame modes.
    //
    // The width must beat the LARGEST pool any backend has, not the one this
    // host happens to select: AVX-512 allocates 22 registers. Twelve products
    // sufficed when every pool was six and silently stopped proving anything
    // when the pools grew, which is what the assertion below now catches.
    let mut a = ExprBuilder::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let mut products = Vec::new();
    for k in 0..48 {
        let c = a.push_const(1.0 + k as f32 * 0.37);
        let xk = a.push_binary(OpKind::Add, x, c);
        let yk = a.push_binary(OpKind::Mul, y, c);
        products.push(a.push_binary(OpKind::Mul, xk, yk));
    }
    // Balanced tree so many products stay live at once.
    while products.len() > 1 {
        let mut next = Vec::new();
        for pair in products.chunks(2) {
            next.push(match pair {
                [l, r] => a.push_binary(OpKind::Add, *l, *r),
                [l] => *l,
                _ => unreachable!(),
            });
        }
        products = next;
    }
    let root = products[0];
    let built = a.finish(&[root]);
    let t = Term::new(built.0.entry(), &built.1);

    let res = assert_collapse_matches_interpreter(t, &[], 3, "spill");
    assert!(
        res.spill_count > 0,
        "pressure kernel did not spill (budget grew?) — the scenario proves nothing"
    );
}
