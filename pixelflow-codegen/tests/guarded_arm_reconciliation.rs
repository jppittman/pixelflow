//! A skipped `Select` arm is still a path, and the location table is what
//! every path after the join agrees on.
//!
//! The short-circuit guard branches around an arm whose mask is uniform. The
//! allocator, meanwhile, may schedule a *reconciliation* at the arm's end
//! point — a value coming back into a register for the code that follows the
//! arm. That reload belongs to both paths. Emitting it before patching the
//! branch put it inside the skipped region, so the uniform-mask path arrived
//! with the register unloaded while the emitter's table claimed the value was
//! in it, and every later read of that value read whatever the register
//! happened to hold.
//!
//! Nothing catches this except running the guarded path: the placement is
//! self-consistent, the arms are correct, and only the *ordering* of two
//! emissions at one index is wrong. So this sweeps a family of kernels that
//! spill a value across a guarded arm and read it after, and runs each under
//! masks that take the arm, skip it, and mix — against the scalar oracle.
//!
//! The second sweep here is about the *floor* rather than the ordering. Every
//! register an instruction needs is a reservation the allocator makes, and
//! inside a guarded arm the earlier attempt at that found a residue: values
//! that could be neither split nor spilled whole, leaving an instruction with
//! nowhere to put its scratch. Splitting removed the residue — a value's store
//! goes at its **definition**, which a guard cannot skip without skipping every
//! read of it, so *any* resident value is a legal eviction inside an arm — but
//! "removed" is an argument, and this is the sweep that holds it: nested
//! guarded `Select`s over a pool at `MIN_SCRATCH`, deep enough that every
//! operand inside the innermost arm has been evicted, against the oracle under
//! every combination of uniform and mixed masks.

use pixelflow_codegen::emit::EmitCtx;
use pixelflow_codegen::{CompiledKernel, Point4};
use pixelflow_ir::expr::{Environment, ExprBuilder, ExprData, ExprRef};
use pixelflow_ir::{BindingTable, LatticeShape, OpKind, Rooted, Term, eval_scalar};

const LANES: usize = pixelflow_codegen::JIT_VECTOR_BYTES / 4;

/// A kernel that spills `X·Y` under pressure, guards a `Select` arm, and reads
/// the spilled value `after_reads` times *following* the arm — which is where
/// the allocator puts the reload that the skipped path used to jump over.
///
/// `width` sets how many independent live values compete for the pool, so the
/// sweep covers schedules that spill in different places rather than one.
fn kernel(width: u32, after_reads: usize) -> (Rooted<ExprData>, Environment) {
    let mut a = ExprBuilder::new();
    let x = a.push_var(0);
    let y = a.push_var(1);

    // The mask is Y's sign, so a caller makes it uniform per batch by
    // choosing the signs of the row it samples.
    let zero = a.push_const(0.0);
    let cond = a.push_binary(OpKind::Gt, y, zero);
    // Computed first, read last: what eviction takes when the filler fills
    // the pool.
    let split = a.push_binary(OpKind::Mul, x, y);

    let terms: Vec<ExprRef> = (1..=width)
        .map(|i| {
            let c = a.push_const(i as f32);
            a.push_binary(OpKind::Add, x, c)
        })
        .collect();
    let mid = terms[1..]
        .iter()
        .fold(terms[0], |acc, &t| a.push_binary(OpKind::Add, acc, t));

    // One arm only: the false side is `split`, computed outside the `Select`,
    // so the true arm's end IS the `Select`'s own index and lies outside every
    // arm. The `Select` reads `split`, which by then is spilled, and `split` is
    // read again after — so the allocator brings it back into a pool register
    // exactly at the join. That reload is the one the skipping path must not
    // jump over.
    let base = a.push_binary(OpKind::Mul, mid, y);
    let t1 = a.push_binary(OpKind::Mul, base, base);
    let t2 = a.push_binary(OpKind::Add, t1, base);
    let t3 = a.push_binary(OpKind::Mul, t2, t1);
    let sel = a.push_ternary(OpKind::Select, cond, t3, split);

    let mut acc = sel;
    for _ in 0..after_reads {
        acc = a.push_binary(OpKind::Add, acc, split);
        acc = a.push_binary(OpKind::Mul, acc, split);
    }
    a.finish(&[acc])
}

/// Y's sign decides the mask: all-negative skips the true arm, all-positive
/// skips the false arm, and the mixed row takes neither branch. The magnitude
/// still varies per lane, so the arms are not computing one value.
fn lanes_for(kind: usize) -> [f32; LANES] {
    core::array::from_fn(|i| {
        let sign = match kind {
            0 => -1.0,
            1 => 1.0,
            _ if i % 2 == 0 => 1.0,
            _ => -1.0,
        };
        sign * (1.5 + i as f32 * 0.25)
    })
}

#[test]
fn a_reload_at_a_guarded_arms_end_happens_on_the_skipping_path_too() {
    let bindings = BindingTable::empty();
    let mut checked = 0usize;
    for pool in [6u8, 7, 8, 10] {
        for width in 4u32..24 {
            for after_reads in 1..=4 {
                let built = kernel(width, after_reads);
                let t = Term::new(built.0.entry(), &built.1);
                let compiled = EmitCtx::with_max_regs(pool).compile(t).expect("compiles");
                let jit = CompiledKernel::new(compiled.code, LatticeShape::POINT);
                for kind in 0..3 {
                    let xs: [f32; LANES] = core::array::from_fn(|i| 0.5 + i as f32);
                    let ys = lanes_for(kind);
                    let zero = [0.0f32; LANES];
                    let got: [f32; LANES] = unsafe { jit.call(Point4::new(xs, ys, zero, zero)) };
                    for lane in 0..LANES {
                        let point = [xs[lane], ys[lane]];
                        let want = eval_scalar(t, &point, &bindings);
                        assert_eq!(
                            got[lane].to_bits(),
                            want.to_bits(),
                            "pool {pool}, width {width}, {after_reads} reads after the arm, \
                         mask kind {kind}, lane {lane}: JIT {} vs oracle {want}",
                            got[lane],
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 0, "the sweep verified nothing");
}

/// Nested guarded `Select`s whose arms are long enough that the pool is empty
/// inside them.
///
/// `width` independent products are computed before the selects and read after
/// them, so they are live across the whole nest and every register is spoken
/// for; each arm then computes a chain used only by itself, which is what makes
/// it exclusive and so guardable. An instruction inside the inner arm therefore
/// reserves its scratch with nothing free — the case the residue was about.
///
/// X's and Y's signs carry the two masks, so a caller makes either uniform by
/// choosing the signs of the batch it samples.
fn nested_guards(width: u32, depth: usize) -> (Rooted<ExprData>, Environment) {
    let mut a = ExprBuilder::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let zero = a.push_const(0.0);

    let outer_mask = a.push_binary(OpKind::Gt, x, zero);
    let inner_mask = a.push_binary(OpKind::Gt, y, zero);

    // Live across everything below: computed first, read last.
    let fillers: Vec<ExprRef> = (0..width)
        .map(|i| {
            let c = a.push_const(i as f32 + 1.25);
            a.push_binary(OpKind::Mul, x, c)
        })
        .collect();

    // A chain nothing outside its own arm reads.
    let chain = |a: &mut ExprBuilder, seed: ExprRef, other: ExprRef| {
        let mut v = a.push_binary(OpKind::Mul, seed, other);
        for k in 0..depth {
            let c = a.push_const(k as f32 + 0.5);
            v = a.push_binary(OpKind::Add, v, c);
            // Mul then Add, never `MulAdd`: the fused form rounds once with
            // `+fma` and twice without, so an oracle comparison over it is a
            // test of the build's target features rather than of allocation.
            let m = a.push_binary(OpKind::Mul, v, seed);
            v = a.push_binary(OpKind::Add, m, other);
        }
        v
    };

    let inner_true = chain(&mut a, y, x);
    let inner_false = chain(&mut a, x, y);
    let inner = a.push_ternary(OpKind::Select, inner_mask, inner_true, inner_false);

    let outer_true = chain(&mut a, inner, y);
    let outer_false = chain(&mut a, x, y);
    let outer = a.push_ternary(OpKind::Select, outer_mask, outer_true, outer_false);

    let mut acc = outer;
    for f in fillers {
        acc = a.push_binary(OpKind::Add, acc, f);
    }
    a.finish(&[acc])
}

/// Uniform-negative, uniform-positive, or alternating — as a sign on a
/// magnitude that still varies per lane, so the arms compute different things
/// in different lanes.
fn mask_lanes(kind: usize, magnitude: impl Fn(usize) -> f32) -> [f32; LANES] {
    core::array::from_fn(|i| {
        let sign = match kind {
            0 => -1.0,
            1 => 1.0,
            _ if i % 2 == 0 => 1.0,
            _ => -1.0,
        };
        sign * magnitude(i)
    })
}

/// Every reservation inside a guarded arm has a candidate, and the code it
/// produces is right on all three paths.
///
/// The floor argument this pins: at any instruction the pool has at least
/// `MIN_SCRATCH` registers, an instruction claims at most that many roles, and
/// each remaining slot is free or holds a value that can be **split** — the
/// loser keeps the register it held up to that point and continues in its slot,
/// whose store was emitted at its definition. A guard skips a definition only
/// by skipping every read of it, so that store is on every path that reads the
/// value, inside a guarded arm exactly as outside one. Nothing is unevictable,
/// so nothing can run out.
#[test]
fn a_reservation_inside_a_nested_guarded_arm_always_has_a_register() {
    let bindings = BindingTable::empty();
    let floor = pixelflow_codegen::emit::regalloc::RegisterFile::MIN_SCRATCH;
    let mut checked = 0usize;
    for pool in [floor, floor + 1, floor + 3] {
        for width in [2u32, 6, 12] {
            for depth in [1usize, 3, 6] {
                let built = nested_guards(width, depth);
                let t = Term::new(built.0.entry(), &built.1);
                let compiled = EmitCtx::with_max_regs(pool).compile(t).expect("compiles");
                let jit = CompiledKernel::new(compiled.code, LatticeShape::POINT);
                for outer in 0..3 {
                    for inner in 0..3 {
                        let xs = mask_lanes(outer, |i| 0.75 + i as f32 * 0.5);
                        let ys = mask_lanes(inner, |i| 1.25 + i as f32 * 0.125);
                        let zero = [0.0f32; LANES];
                        let got: [f32; LANES] =
                            unsafe { jit.call(Point4::new(xs, ys, zero, zero)) };
                        for lane in 0..LANES {
                            let point = [xs[lane], ys[lane]];
                            let want = eval_scalar(t, &point, &bindings);
                            assert_eq!(
                                got[lane].to_bits(),
                                want.to_bits(),
                                "pool {pool}, width {width}, depth {depth}, masks \
                                 ({outer}, {inner}), lane {lane}: JIT {} vs oracle {want}",
                                got[lane],
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 0, "the sweep verified nothing");
}
