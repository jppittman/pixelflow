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
//!
//! **Sibling folds.** Every row above is point-shaped, with no `Reduce`
//! surviving, so the whole of what a loop nest does (a fold's slots, the
//! scopes it opens, the roots parked for them, a branch inside a loop body)
//! never reaches the emitter under them, and a refactor of scoping,
//! allocation, frames or labels could move every one of those and leave all
//! nine lines identical. The rows after them are the ones that can see it:
//! kernels with a surviving `Reduce` that varies with the column, compiled
//! at a width whose remainder makes the column fold two sibling folds that
//! share one `Reduce`. Each prints the same line the old rows do and then
//! the rest of what the compile reports: `CompileResult`'s counts and every
//! scope's `EmitTraffic`, indented under it. Never a wall time: the output
//! is diffed.
//!
//! The kernels are `tests/support/sibling_rows.rs`, the file the byte
//! golden in `emit::tests::sibling_folds` includes, so the two measure one
//! definition of each row. The golden pins all three backends from any
//! host; this prints the host's own, through the production path as well.

use pixelflow_codegen::emit::{CompileResult, ScopeTraffic, compile};
use pixelflow_codegen::jit_cache::{self, Linked};
use pixelflow_codegen::{CompileError, fnv1a64};
use pixelflow_ir::{ExprArena, ExprId, Kernel, LatticeShape, OpKind, Uniform};

/// The kernels that reach sibling folds.
mod sibling_rows {
    include!("../tests/support/sibling_rows.rs");
}

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

/// The production compile — optimize, link, emit — or why there is none.
fn through_the_jit(
    arena: &ExprArena,
    root: ExprId,
    shape: LatticeShape,
) -> Result<Linked, CompileError> {
    jit_cache::compile(&Kernel::from_parts(arena.clone(), root), shape)
}

/// The production compile's bytes, or why there are none.
fn jit_columns(production: &Result<Linked, CompileError>) -> String {
    match production {
        Ok(linked) => {
            let bytes = linked.kernel.as_bytes();
            format!("jit_len={:<6} jit_fnv={:016x}", bytes.len(), fnv1a64(bytes))
        }
        Err(e) => format!("jit ERROR {e:?}"),
    }
}

/// What [`compile`] alone reports of a kernel, in the columns every row
/// shares — or why it reports nothing.
fn emitter_columns(compiled: &Result<CompileResult, CompileError>) -> String {
    match compiled {
        Ok(r) => {
            let bytes = r.code.as_bytes();
            format!(
                "len={:<6} fnv={:016x} spills={} hoisted={}",
                bytes.len(),
                fnv1a64(bytes),
                r.spill_count,
                r.hoisted_values
            )
        }
        Err(e) => format!("ERROR {e:?}"),
    }
}

/// A sibling-fold kernel and the lattice it is compiled over.
struct SiblingRow {
    name: &'static str,
    arena: ExprArena,
    root: ExprId,
    shape: LatticeShape,
}

/// The rows that reach sibling folds, named as the golden names them.
///
/// `wL` is one batch of the host's own lane count, so a run here and a run
/// on the other tier differ in which rows have a remainder fold and in the
/// bytes; the names do not.
fn sibling_cases() -> Vec<SiblingRow> {
    use pixelflow_codegen::jit_vector_bytes;
    use sibling_rows::{REMAINDER_WIDTH, ROWS};
    const BYTES_PER_LANE: usize = 4;

    let one_batch = (jit_vector_bytes() / BYTES_PER_LANE) as u32;
    let row = |name, (arena, root): (ExprArena, ExprId), columns| SiblingRow {
        name,
        arena,
        root,
        shape: LatticeShape::new([columns, ROWS]),
    };
    vec![
        row("glyph_like_w1", sibling_rows::glyph_like(), 1),
        row("glyph_like_wL", sibling_rows::glyph_like(), one_batch),
        row(
            "glyph_like_w37",
            sibling_rows::glyph_like(),
            REMAINDER_WIDTH,
        ),
        row(
            "two_sibling_folds_w37",
            sibling_rows::two_sibling_folds(),
            REMAINDER_WIDTH,
        ),
        row(
            "parked_roots_w37",
            sibling_rows::parked_roots(sibling_rows::PARKED_TERMS),
            REMAINDER_WIDTH,
        ),
        row(
            "guarded_if_in_fold_w37",
            sibling_rows::guarded_if_in_fold(),
            REMAINDER_WIDTH,
        ),
    ]
}

/// One scope's counts, in a fixed order.
fn counts(t: &ScopeTraffic) -> String {
    format!(
        "instructions={} loads_transient={} loads_kept={} remats={} stores={} bytes={}",
        t.instructions, t.loads_transient, t.loads_kept, t.remats, t.stores, t.bytes
    )
}

/// A sibling row: the old rows' line, then everything a compile reports.
fn print_sibling_row(row: &SiblingRow) {
    let SiblingRow {
        name,
        arena,
        root,
        shape,
    } = row;
    let production = jit_columns(&through_the_jit(arena, *root, *shape));
    let compiled = compile(arena, *root, *shape);
    println!("{name:<24} {} {production}", emitter_columns(&compiled));
    let Ok(r) = compiled else {
        return;
    };
    let t = &r.traffic;
    println!(
        "    compile: spill_count={} spill_bytes={} hoisted_values={} vector_bytes={} pool={}",
        r.spill_count, r.spill_bytes, r.hoisted_values, t.vector_bytes, t.pool
    );
    println!(
        "    traffic: carried={} scopes={}",
        t.carried,
        t.scopes.len()
    );
    println!("    scaffold: {}", counts(&t.scaffold));
    for (scope, (traffic, trips)) in t.scopes.iter().zip(&t.trips).enumerate() {
        println!("    scope {scope}: trips={trips} {}", counts(traffic));
    }
}

fn main() {
    // An error is printed rather than propagated: a kernel this cannot
    // compile is still a data point, and the other rows are still worth
    // having.
    for (name, arena, root) in cases() {
        let jit = jit_columns(&through_the_jit(&arena, root, LatticeShape::POINT));
        let compiled = compile(&arena, root, LatticeShape::POINT);
        println!("{name:<14} {} {jit}", emitter_columns(&compiled));
    }
    for row in sibling_cases() {
        print_sibling_row(&row);
    }
}
