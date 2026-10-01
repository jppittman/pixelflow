//! Did this change move any bytes?
//!
//! Compiles a spread of kernels and prints one line per kernel: emitted
//! length, a hash of the bytes, and the two facts about the loop nest a
//! refactor is most likely to disturb — how much spilled, and how much was
//! hoisted out of the loop.
//!
//! It exists because "byte-identical" is the honest gate for a *refactor*, and
//! nothing else in the tree answers it. The test suite answers "still
//! correct", which is weaker: a rewrite that quietly costs eight registers is
//! green. Diffing two runs of this catches that, and catches the reverse
//! mistake too — claiming a refactor is byte-identical when it is not.
//!
//! ```sh
//! git worktree add /tmp/base <the-revision-before> --detach
//! cargo run -p pixelflow-codegen --example byte_probe > /tmp/after.txt
//! (cd /tmp/base && cargo run -p pixelflow-codegen --example byte_probe) > /tmp/before.txt
//! diff /tmp/before.txt /tmp/after.txt
//! ```
//!
//! Not a test, and deliberately: a pinned hash would fail on every
//! *intentional* change, which is most of them, and a gate that cries wolf
//! gets deleted. This is a question you ask, not an alarm that watches.
//!
//! **One host, one ISA.** `compile` emits for the machine it runs on, so a
//! run here says nothing about the other backends; `cargo xtask isa-matrix`
//! is what covers those.
//!
//! **Two columns per kernel.** The first is [`compile`] alone: the emitter
//! over the arena as written. The second, `jit_*`, is
//! [`jit_cache::compile`](pixelflow_codegen::jit_cache::compile), the
//! production path — optimize, link, emit — which is where a change to the
//! optimizer or the linker shows up and the first column cannot see it. The
//! two `named_*` kernels hold a reference (`Kernel::by_ref`), so they are
//! the rows a change to how references are optimized is expected to move.

use pixelflow_codegen::emit::compile;
use pixelflow_codegen::fnv1a64;
use pixelflow_codegen::jit_cache;
use pixelflow_ir::{ExprArena, ExprId, Kernel, LatticeShape, OpKind, Uniform};

fn xy(a: &mut ExprArena) -> (ExprId, ExprId) {
    (a.push_var(0), a.push_var(1))
}

/// `sqrt(x*x + y*y)`
fn radius(a: &mut ExprArena, x: ExprId, y: ExprId) -> ExprId {
    let xx = a.push_binary(OpKind::Mul, x, x);
    let yy = a.push_binary(OpKind::Mul, y, y);
    let d = a.push_binary(OpKind::Add, xx, yy);
    a.push_unary(OpKind::Sqrt, d)
}

/// One kernel per thing the emitter does differently, since a probe only
/// covers what it reaches.
fn cases() -> Vec<(&'static str, ExprArena, ExprId)> {
    let mut out = Vec::new();

    // The smallest thing that is still a collapse: the scaffold, alone.
    {
        let mut a = ExprArena::new();
        let (x, y) = xy(&mut a);
        let r = a.push_binary(OpKind::Add, x, y);
        out.push(("add_xy", a, r));
    }

    // Transcendentals and constants — the polynomial expansions, and the
    // constant pool on a backend that has one.
    {
        let mut a = ExprArena::new();
        let (x, y) = xy(&mut a);
        let s = radius(&mut a, x, y);
        let k = a.push_const(3.7);
        let m = a.push_binary(OpKind::Mul, s, k);
        let sn = a.push_unary(OpKind::Sin, m);
        let amp = a.push_const(0.5);
        let p = a.push_binary(OpKind::Mul, sn, amp);
        let b = a.push_const(0.25);
        let r = a.push_binary(OpKind::Add, p, b);
        out.push(("swirl", a, r));
    }

    // Y-only: a root hoisted to the per-row region.
    {
        let mut a = ExprArena::new();
        let (x, y) = xy(&mut a);
        let ky = a.push_const(1.7);
        let sy = a.push_binary(OpKind::Mul, y, ky);
        let row = a.push_unary(OpKind::Sin, sy);
        let row2 = a.push_unary(OpKind::Exp, row);
        let r = a.push_binary(OpKind::Add, x, row2);
        out.push(("row_invariant", a, r));
    }

    // Constant-only under a Y-only: both regions non-empty, which is the
    // case that exercises a park being read by a scope two levels in.
    {
        let mut a = ExprArena::new();
        let (x, y) = xy(&mut a);
        let c = a.push_const(0.31);
        let call = a.push_unary(OpKind::Sin, c);
        let call2 = a.push_unary(OpKind::Exp, call);
        let sy = a.push_binary(OpKind::Mul, y, call2);
        let row = a.push_unary(OpKind::Sqrt, sy);
        let r = a.push_binary(OpKind::Add, x, row);
        out.push(("both_regions", a, r));
    }

    // An If whose mask varies by lane, with enough work in an arm to be
    // worth a guard: the branch path.
    {
        let mut a = ExprArena::new();
        let (x, y) = xy(&mut a);
        let s = radius(&mut a, x, y);
        let one = a.push_const(1.0);
        let m = a.push_binary(OpKind::Lt, s, one);
        let hot = a.push_unary(OpKind::Sin, s);
        let hot2 = a.push_unary(OpKind::Exp, hot);
        let cold = a.push_unary(OpKind::Sqrt, s);
        let r = a.push_ternary(OpKind::If, m, hot2, cold);
        out.push(("if_guard", a, r));
    }

    // More live values than the pool holds: eviction, splitting, reloads.
    {
        let mut a = ExprArena::new();
        let (x, y) = xy(&mut a);
        let mut terms = Vec::new();
        for i in 0..24 {
            let k = a.push_const(i as f32 * 0.37 + 0.1);
            let t = a.push_binary(OpKind::Mul, x, k);
            let u = a.push_binary(OpKind::Add, t, y);
            terms.push(a.push_unary(OpKind::Sin, u));
        }
        let mut acc = terms[0];
        for t in &terms[1..] {
            acc = a.push_binary(OpKind::Add, acc, *t);
        }
        out.push(("wide_spill", a, acc));
    }

    // A uniform: the link step, where the code is compiled against dense
    // slots and the caller's identities are mapped onto them.
    {
        let tint = Uniform::new(0.75).kernel();
        let k = Kernel::x().mul(&tint).add(&Kernel::y());
        let (a, r) = k.parts();
        out.push(("uniform_tint", a.clone(), r));
    }

    // A named kernel beside other work: a reference the optimizer sees.
    {
        let body = Kernel::x()
            .mul(&Kernel::x())
            .add(&Kernel::y().mul(&Kernel::y()))
            .sqrt()
            .mul(&Kernel::constant(3.7))
            .sin();
        let k = body.by_ref().add(&Kernel::y().mul(&Kernel::constant(2.0)));
        let (a, r) = k.parts();
        out.push(("named_beside", a.clone(), r));
    }

    // Two named kernels as the arms of a choice over a uniform: a font's id
    // tree, one level deep.
    {
        let id = Uniform::new(1.0).kernel();
        let lower = Kernel::x()
            .mul(&Kernel::constant(0.5))
            .add(&Kernel::y())
            .by_ref();
        let upper = Kernel::y().mul(&Kernel::y()).sub(&Kernel::x()).by_ref();
        let k = id.lt(&Kernel::constant(1.0)).select(&lower, &upper);
        let (a, r) = k.parts();
        out.push(("named_arms", a.clone(), r));
    }

    out
}

/// The production compile's bytes, or why there are none.
fn through_the_jit(arena: &ExprArena, root: ExprId) -> String {
    let kernel = Kernel::from_parts(arena.clone(), root);
    match jit_cache::compile(&kernel, LatticeShape::POINT) {
        Ok(linked) => {
            let bytes = linked.kernel.code_bytes();
            format!("jit_len={:<6} jit_fnv={:016x}", bytes.len(), fnv1a64(bytes))
        }
        Err(e) => format!("jit ERROR {e:?}"),
    }
}

fn main() {
    for (name, arena, root) in cases() {
        let jit = through_the_jit(&arena, root);
        match compile(&arena, root, LatticeShape::POINT) {
            Ok(r) => {
                let bytes = r.code.as_bytes();
                println!(
                    "{name:<14} len={:<6} fnv={:016x} spills={} hoisted={} {jit}",
                    bytes.len(),
                    fnv1a64(bytes),
                    r.spill_count,
                    r.hoisted_values
                );
            }
            // Printed rather than propagated: a kernel this cannot compile is
            // still a data point, and the other rows are still worth having.
            Err(e) => println!("{name:<14} ERROR {e:?} {jit}"),
        }
    }
}
