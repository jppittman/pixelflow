//! The integer-domain ops compute what the integers say, on the JIT.
//!
//! `TruncToInt`, `IntToFloat`, `IAdd`, `Shl`, `Shr`, `BitAnd` and `BitOr` are
//! the primitives `expand_transcendentals` lowers exp and log to, and the ones
//! `Bits` exposes. A lane holds an f32 bit pattern; these ops read it as an
//! `i32` or `u32` and write one back, so the contract is the integer's, not the
//! float's: every lane must match bit for bit.
//!
//! The oracle is Rust's integer and float arithmetic (`trunc() as i32`,
//! `checked_add`, `u32` shifts and bit operators, `i as f32` for |i| < 2^24),
//! which shares no code with the emitter. Operands arrive as gathers over
//! buffers, as in `eq_ne_jit.rs`: a gather loads the pattern unchanged, where a
//! lattice coordinate is float arithmetic and would turn `-0.0` into `+0.0`.
//! Every lane is a runtime value the emitter cannot fold away.
//!
//! Each kernel runs twice: straight from the arena to the emitter, and through
//! the optimizing entry (`jit_cache::compile`) that every production kernel
//! takes. The optimizer may not change one bit of these answers.
//!
//! The answers are the x86 row of CLAUDE.md "Floating point at the edges":
//! `cvttps2dq` truncates toward zero, `vpslld`/`vpsrld` zero-fill and `Shr` is
//! logical. Every input here is in range, where the aarch64 row agrees too; its
//! column is not written here because it cannot run on this host. Counts outside
//! `0..32` are target-specific and are not tested.

#![cfg(target_arch = "x86_64")]

use std::sync::Arc;

use pixelflow_codegen::emit::{CompiledKernel, compile};
use pixelflow_codegen::jit_cache;
use pixelflow_ir::arena::{BufferDecl, BufferIdentity};
use pixelflow_ir::{ExprArena, ExprId, Kernel, LatticeShape, OpKind};

/// Lanes per row: deliberately not a multiple of any tier's batch, so every
/// row ends in a remainder.
const WIDTH: usize = 37;

/// Both routes a kernel is compiled by: the emitter alone, and the optimizing
/// entry that runs the e-graph first.
#[derive(Clone, Copy, Debug)]
enum Route {
    Emitted,
    Optimized,
}

const ROUTES: [Route; 2] = [Route::Emitted, Route::Optimized];

/// Compile the kernel `build` makes by `route`, over `operands` gathered
/// buffers, and return each output lane's bit pattern. Operand `k` of lane `i`
/// is `inputs[i].0` for `k == 0` and `inputs[i].1` for `k == 1`.
fn evaluate(
    route: Route,
    operands: usize,
    build: impl Fn(&mut ExprArena, &[ExprId]) -> ExprId,
    inputs: &[(u32, u32)],
) -> Vec<u32> {
    let rows = inputs.len().div_ceil(WIDTH);
    let mut padded = inputs.to_vec();
    padded.resize(rows * WIDTH, (0, 0));
    let lanes: Vec<Vec<f32>> = (0..operands)
        .map(|k| {
            padded
                .iter()
                .map(|p| f32::from_bits(if k == 0 { p.0 } else { p.1 }))
                .collect()
        })
        .collect();
    let ids: Vec<BufferIdentity> = (0..operands).map(|_| BufferIdentity::mint()).collect();

    let mut arena = ExprArena::new();
    let x = arena.push_var(0);
    let y = arena.push_var(1);
    let gathers: Vec<ExprId> = ids
        .iter()
        .map(|&id| {
            let buffer = arena.declare_buffer(BufferDecl {
                id,
                width: WIDTH as u32,
                height: rows as u32,
            });
            arena.push_gather(buffer, x, y)
        })
        .collect();
    let root = build(&mut arena, &gathers);

    let shape = LatticeShape::new([WIDTH as u32, rows as u32]);
    // Buffers in the order the kernel binds them, then the uniform block, then
    // the origin. The optimizing entry may reorder or drop buffers, so its
    // binding comes from the link, not from the declaration order.
    let (jit, mut ctx): (Arc<CompiledKernel>, Vec<*const f32>) = match route {
        Route::Emitted => (
            Arc::new(
                compile(&arena, root, shape)
                    .expect("the kernel compiles")
                    .code,
            ),
            lanes.iter().map(|lane| lane.as_ptr()).collect(),
        ),
        Route::Optimized => {
            let linked = jit_cache::compile(&Kernel::from_parts(arena, root), shape)
                .expect("the optimized kernel compiles");
            let slots = linked
                .buffers
                .iter()
                .map(|decl| {
                    let k = ids
                        .iter()
                        .position(|&id| id == decl.id)
                        .expect("the link names a buffer this kernel declared");
                    lanes[k].as_ptr()
                })
                .collect();
            (linked.kernel, slots)
        }
    };
    let origin = [0.0f32, 0.0];
    let no_uniforms: [f32; 0] = [];
    ctx.push(no_uniforms.as_ptr());
    ctx.push(origin.as_ptr());

    let mut out = vec![f32::NAN; rows * WIDTH];
    // SAFETY: each buffer holds `rows * WIDTH` samples and is read at the
    // lattice's own positions; the uniform block is never read (none is
    // declared); the origin is two floats; `out` holds `rows * WIDTH` with
    // pitch `WIDTH`.
    unsafe { jit.call(ctx.as_ptr(), out.as_mut_ptr(), WIDTH) };
    out.truncate(inputs.len());
    out.into_iter().map(f32::to_bits).collect()
}

/// Every route's output must equal the integer oracle's bits for every input.
fn check(
    name: &str,
    operands: usize,
    build: impl Fn(&mut ExprArena, &[ExprId]) -> ExprId + Copy,
    inputs: &[(u32, u32)],
    oracle: impl Fn(u32, u32) -> u32,
) {
    for route in ROUTES {
        let got = evaluate(route, operands, build, inputs);
        for (&(x, y), got) in inputs.iter().zip(got) {
            let want = oracle(x, y);
            assert_eq!(
                got, want,
                "{name} on {route:?}: x={x:#010x}, y={y:#010x}: got {got:#010x}, want {want:#010x}",
            );
        }
    }
}

/// A unary op over operand 0.
fn unary(op: OpKind) -> impl Fn(&mut ExprArena, &[ExprId]) -> ExprId + Copy {
    move |arena, ops| arena.push_unary(op, ops[0])
}

/// A binary op over operands 0 and 1.
fn binary(op: OpKind) -> impl Fn(&mut ExprArena, &[ExprId]) -> ExprId + Copy {
    move |arena, ops| arena.push_binary(op, ops[0], ops[1])
}

/// Operand 0 shifted by the `Const` `count`, as the arena spells it.
fn shift_by(op: OpKind, count: u32) -> impl Fn(&mut ExprArena, &[ExprId]) -> ExprId + Copy {
    move |arena, ops| {
        let c = arena.push_const(count as f32);
        arena.push_binary(op, ops[0], c)
    }
}

/// xorshift64: a fixed sequence, so every run checks the same inputs.
fn xorshift(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 32) as u32
}

const SEED: u64 = 0x9e37_79b9_7f4a_7c15;
const SAMPLES: usize = 512;

// ── TruncToInt: truncation toward zero, in range ─────────────────────────────

#[test]
fn trunc_to_int_truncates_toward_zero() {
    const SPECIALS: &[f32] = &[
        0.0,
        -0.0,
        0.25,
        -0.25,
        0.5,
        -0.5,
        0.999_999,
        -0.999_999,
        1.0,
        -1.0,
        1.5,
        -1.5,
        1.9,
        -1.9,
        2.5,
        -2.5,
        7.75,
        -7.75,
        123.456,
        -123.456,
        65_535.75,
        -65_535.75,
        16_777_216.0,
        -16_777_216.0,
        1_000_000_000.0,
        -1_000_000_000.0,
        2_147_483_520.0,
        -2_147_483_520.0,
        // i32::MIN is itself representable, so it is in range.
        -2_147_483_648.0,
    ];
    let mut inputs: Vec<(u32, u32)> = SPECIALS.iter().map(|x| (x.to_bits(), 0)).collect();

    // Random bit patterns, kept where the value is an i32 in range; then
    // fixed-point values with fractions at every magnitude up to 2^23.
    let mut state = SEED;
    let mut random = 0;
    while random < SAMPLES {
        let x = f32::from_bits(xorshift(&mut state));
        if (-2_147_483_648.0..2_147_483_648.0).contains(&x) {
            inputs.push((x.to_bits(), 0));
            random += 1;
        }
    }
    for _ in 0..SAMPLES {
        let x = ((xorshift(&mut state) as i32) >> 4) as f32 / 16.0;
        inputs.push((x.to_bits(), 0));
    }

    for &(x, _) in &inputs {
        let x = f32::from_bits(x);
        assert!(
            (-2_147_483_648.0..2_147_483_648.0).contains(&x),
            "the inputs are in i32 range: {x}"
        );
    }
    check(
        "trunc_to_int",
        1,
        unary(OpKind::TruncToInt),
        &inputs,
        |x, _| (f32::from_bits(x).trunc() as i32) as u32,
    );
}

// ── IntToFloat: the lane's bits read as an i32, exact below 2^24 ─────────────

#[test]
fn int_to_float_reads_the_bits_as_an_integer() {
    const SPECIALS: &[i32] = &[
        0,
        1,
        -1,
        2,
        -2,
        3,
        255,
        -255,
        256,
        -256,
        65_535,
        12_345_678,
        -12_345_678,
        1 << 23,
        -(1 << 23),
        (1 << 24) - 1,
        -((1 << 24) - 1),
    ];
    let mut inputs: Vec<(u32, u32)> = SPECIALS.iter().map(|&i| (i as u32, 0)).collect();

    // Random integers with |i| < 2^24, the range f32 holds exactly.
    let mut state = SEED;
    for _ in 0..SAMPLES {
        let i = (xorshift(&mut state) as i32) >> 8;
        inputs.push((i as u32, 0));
    }

    check(
        "int_to_float",
        1,
        unary(OpKind::IntToFloat),
        &inputs,
        |x, _| {
            let i = x as i32;
            assert!(i.unsigned_abs() < 1 << 24, "{i} is not exact in f32");
            (i as f32).to_bits()
        },
    );
}

// ── Bits / Float: a reinterpretation is its operand, bit for bit ─────────────

/// Every kind of pattern a lane can hold, including the ones a *number* would
/// not survive a round trip through: both NaN signs, both infinities, `-0.0`,
/// all-ones, a denormal, and the bit patterns `expand_exp2`/`expand_log2`
/// build their exponent fields from.
fn every_kind_of_pattern() -> Vec<(u32, u32)> {
    const SPECIALS: &[u32] = &[
        0x0000_0000,
        0x8000_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC0_0000,
        0xFFC0_0000,
        0xFFFF_FFFF,
        0x0000_0001,
        0x007F_FFFF,
        0x7F7F_FFFF,
        127,
        23,
    ];
    let mut inputs: Vec<(u32, u32)> = SPECIALS.iter().map(|&b| (b, 0)).collect();
    let mut state = SEED;
    for _ in 0..SAMPLES {
        inputs.push((xorshift(&mut state), 0));
    }
    inputs
}

/// A `Bits` or `Float` node is its operand: the same 32 bits come out, for
/// every pattern, on both routes. This observes the value and not the
/// instruction count (docs/process/2026-10-13-tests-check-the-as-if-contract.md);
/// that the node costs no instruction is `byte_probe`'s to say.
#[test]
fn bits_and_float_are_their_operand() {
    let inputs = every_kind_of_pattern();
    check("bits", 1, unary(OpKind::Bits), &inputs, |x, _| x);
    check("float", 1, unary(OpKind::Float), &inputs, |x, _| x);
    check(
        "float_of_bits",
        1,
        |arena, ops| {
            let bits = arena.push_unary(OpKind::Bits, ops[0]);
            arena.push_unary(OpKind::Float, bits)
        },
        &inputs,
        |x, _| x,
    );
    check(
        "bits_of_float",
        1,
        |arena, ops| {
            let real = arena.push_unary(OpKind::Float, ops[0]);
            arena.push_unary(OpKind::Bits, real)
        },
        &inputs,
        |x, _| x,
    );
}

/// The shape the expansions use — an integer op between the two
/// reinterpretations — computes exactly what the integer op alone computes:
/// `Float(Shl(Bits(x), k))` is `x << k`, and the same for `Shr`.
#[test]
fn a_reinterpreted_shift_is_the_shift() {
    const COUNT: u32 = 23;
    let inputs = every_kind_of_pattern();
    for (name, op, oracle) in [
        (
            "float_shl_bits",
            OpKind::Shl,
            (|x: u32| x << COUNT) as fn(u32) -> u32,
        ),
        ("float_shr_bits", OpKind::Shr, |x: u32| x >> COUNT),
    ] {
        check(
            name,
            1,
            move |arena, ops| {
                let bits = arena.push_unary(OpKind::Bits, ops[0]);
                let count = arena.push_const(COUNT as f32);
                let shifted = arena.push_binary(op, bits, count);
                arena.push_unary(OpKind::Float, shifted)
            },
            &inputs,
            |x, _| oracle(x),
        );
    }
}

// ── IAdd: integer addition on bit patterns, pairs that do not overflow ───────

#[test]
fn iadd_adds_integers_without_overflow() {
    let specials: &[(i32, i32)] = &[
        (0, 0),
        (1, 1),
        (-1, 1),
        (-1, -1),
        (i32::MAX - 1, 1),
        (i32::MAX, 0),
        (i32::MIN, 0),
        (i32::MIN, 1),
        (i32::MIN, i32::MAX),
        (i32::MIN + 1, -1),
        (1 << 30, -(1 << 30)),
        (-(1 << 30), -(1 << 30)),
        (0x5555_5555, 0x2AAA_AAAA),
    ];
    let mut inputs: Vec<(u32, u32)> = specials
        .iter()
        .map(|&(a, b)| (a as u32, b as u32))
        .collect();

    // Each operand is in [-2^30, 2^30), so every sum fits in an i32.
    let mut state = SEED;
    for _ in 0..SAMPLES {
        let a = (xorshift(&mut state) as i32) >> 1;
        let b = (xorshift(&mut state) as i32) >> 1;
        inputs.push((a as u32, b as u32));
    }

    check("iadd", 2, binary(OpKind::IAdd), &inputs, |x, y| {
        (x as i32)
            .checked_add(y as i32)
            .expect("the pairs are chosen so the sum fits in i32") as u32
    });
}

// ── Shl and Shr: every count in 0..32, including 0 (identity) and 31 ─────────

/// Bit patterns with the sign bit, the mantissa, NaN payloads and a mask set,
/// plus a fixed random sample.
fn shift_inputs() -> Vec<(u32, u32)> {
    const SPECIALS: &[u32] = &[
        0,
        1,
        0x8000_0000,
        0x8000_0001,
        0xFFFF_FFFF,
        0x7FFF_FFFF,
        0xDEAD_BEEF,
        0x3F80_0000,
        0x7FC0_0001,
        0xFFC0_0000,
        0x0080_0000,
        0x007F_FFFF,
    ];
    let mut inputs: Vec<(u32, u32)> = SPECIALS.iter().map(|&x| (x, 0)).collect();
    let mut state = SEED;
    for _ in 0..SAMPLES / 8 {
        inputs.push((xorshift(&mut state), 0));
    }
    inputs
}

#[test]
fn shl_shifts_the_bit_pattern_for_every_count_in_range() {
    let inputs = shift_inputs();
    for count in 0..32u32 {
        check(
            &format!("shl by {count}"),
            1,
            shift_by(OpKind::Shl, count),
            &inputs,
            |x, _| x << count,
        );
    }
}

#[test]
fn shr_is_a_logical_shift_for_every_count_in_range() {
    // Logical, not arithmetic: the sign bit is shifted in as a zero, so a
    // negative pattern stays a large positive one.
    let inputs = shift_inputs();
    for count in 0..32u32 {
        check(
            &format!("shr by {count}"),
            1,
            shift_by(OpKind::Shr, count),
            &inputs,
            |x, _| x >> count,
        );
    }
}

// ── BitAnd and BitOr: bitwise on arbitrary patterns, the all-ones mask ───────

/// Every pair of the special patterns, then a fixed random sample. The
/// all-ones pattern is a comparison mask; `1.0`'s bits are the trap a mask
/// spelled as a number falls into.
fn bit_inputs() -> Vec<(u32, u32)> {
    const SPECIALS: &[u32] = &[
        0,
        0xFFFF_FFFF,
        0x3F80_0000,
        0x0000_0001,
        0x8000_0000,
        0x007F_FFFF,
        0x7FC0_0001,
        0xFFC0_0000,
        0x7F80_0000,
        0x0F0F_0F0F,
    ];
    let mut inputs = Vec::new();
    for &x in SPECIALS {
        for &y in SPECIALS {
            inputs.push((x, y));
        }
    }
    let mut state = SEED;
    for _ in 0..SAMPLES {
        inputs.push((xorshift(&mut state), xorshift(&mut state)));
    }
    inputs
}

#[test]
fn bit_and_is_bitwise_on_arbitrary_patterns() {
    check(
        "bit_and",
        2,
        binary(OpKind::BitAnd),
        &bit_inputs(),
        |x, y| x & y,
    );
}

#[test]
fn bit_or_is_bitwise_on_arbitrary_patterns() {
    check("bit_or", 2, binary(OpKind::BitOr), &bit_inputs(), |x, y| {
        x | y
    });
}
