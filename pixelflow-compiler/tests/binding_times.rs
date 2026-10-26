//! Records, binding times and `Args`
//! (docs/language/2026-09-25-the-language-is-kernel.md §1.3, §1.4; B3).
//!
//! Every `f32` and record parameter of an entry is a **uniform**: an `f32`
//! parameter is one, a record parameter one per field, and a call's values
//! are the kernel's arguments, never its constants. So every call of an entry
//! that takes no kernel is one program — one canonical key, one compiled
//! region — and a program compiled once is rebound per call from the entry's
//! `Args` record, into a block the caller keeps; so is a program the entry's
//! kernel was composed into, beside kernels with no arguments of their own.
//! An entry's count parameter, `n: usize`, is a uniform too: one program for
//! every count, rebound per call, and refused past 2²⁴ where the host writes
//! it.
//!
//! What a record and a count *mean* is pinned against rustc in
//! `rustc_is_the_oracle.rs`; the refusals are `sema`'s and the parser's unit
//! tests. This file pins the binding: keys, positions and rebinding.

use pixelflow_compiler::kernel;
use pixelflow_core::{ArityMismatch, Kernel, Lattice, Manifold, Uniform};
use pixelflow_ir::key::canonical;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

kernel! {
    /// An axis-aligned box.
    pub struct Bounds { pub x0: f32, pub y0: f32, pub x1: f32, pub y1: f32 }

    /// How many terms and rings.
    const N: usize = 4;

    fn inside(b: Bounds, x: f32, y: f32) -> bool {
        (x >= b.x0) & (x <= b.x1) & (y >= b.y0) & (y <= b.y1)
    }

    /// `fg` inside the box, `bg` outside: a record parameter and two scalars.
    pub fn boxed(b: Bounds, fg: f32, bg: f32) -> f32 {
        let c = b;
        if inside(c, X, Y) { fg } else { bg }
    }

    /// A body that never reads its first parameter.
    pub fn skips(unread: f32, a: f32, c: f32) -> f32 { a * 10.0 + c }

    /// The mean of `X·i` over `i ∈ [0, N)`, scaled.
    pub fn ramp(scale: f32) -> f32 {
        (0..N).map(|i| X * (i as f32)).sum::<f32>() * scale / (N as f32)
    }

    /// `N` rings about a box's corner, `fg` on `bg`: the macro doc's
    /// example, compiled, so the doc cannot drift from the language. A
    /// record is its block's: this one could not name `Bounds` from another.
    pub fn rings(b: Bounds, fg: f32, bg: f32) -> f32 {
        let dx = X - b.x0;
        let dy = Y - b.y0;
        let r = (dx * dx + dy * dy).sqrt();
        let within = (X <= b.x1) & (Y <= b.y1);
        let n: f32 = (0..N).map(|i| if r < (i as f32) + 1.0 { 1.0 } else { 0.0 }).sum();
        if within { fg * n / (N as f32) } else { bg }
    }

    /// Σ_{i<n} X·i, its count per call.
    pub fn sum_to(n: usize) -> f32 {
        (0..n).map(|i| X * (i as f32)).sum()
    }

    /// A count between a record and a scalar: declared in parameter order.
    pub fn scaled_sum(b: Bounds, n: usize, s: f32) -> f32 {
        (0..n).map(|i| (X - b.x0) * (i as f32)).sum::<f32>() * s
    }
}

/// The lattice the rebinding tests collapse over: small, and not a whole
/// SIMD batch wide, so a row's remainder is exercised too.
const FRAME: (usize, usize) = (11, 7);

const UNIT: Bounds = Bounds {
    x0: 2.0,
    y0: 1.0,
    x1: 6.0,
    y1: 4.0,
};

/// The key the JIT cache compiles under: the canonical form's shape bytes,
/// which number a kernel's arguments by slot and hold neither their
/// identities nor their values.
fn program_key(k: &Kernel) -> Vec<u8> {
    let (arena, root) = k.parts();
    canonical(arena, root).key
}

/// The address of the code `k` compiles to at `FRAME`'s shape: one compiled
/// region per program, shared by every kernel of it.
fn code_at_frame(k: &Kernel) -> *const u8 {
    let shape = pixelflow_ir::LatticeShape::new([FRAME.0 as u32, FRAME.1 as u32]);
    pixelflow_codegen::jit_cache::compile(k, shape)
        .expect("compile")
        .kernel
        .as_bytes()
        .as_ptr()
}

/// A record parameter is one uniform per field, in field order, before the
/// parameters after it: the call's values, in declaration order.
#[test]
fn a_record_parameter_is_one_uniform_per_field_in_declaration_order() {
    let k = boxed(UNIT, 1.0, 0.25);
    let defaults: Vec<f32> = k.uniforms().iter().map(|u| u.default).collect();
    assert_eq!(defaults, [2.0, 1.0, 6.0, 4.0, 1.0, 0.25]);
    assert_eq!(Lattice::eval_at(&k, 3.0, 2.0), 1.0, "inside the box");
    assert_eq!(Lattice::eval_at(&k, 7.0, 2.0), 0.25, "right of it");
}

/// Every call of this entry is one program: two calls with different values
/// share a canonical key and a compiled region.
#[test]
fn every_call_of_an_entry_is_one_program() {
    let a = boxed(UNIT, 1.0, 0.0);
    let b = boxed(
        Bounds {
            x0: -3.0,
            y0: 0.5,
            x1: 9.0,
            y1: 2.5,
        },
        0.5,
        0.75,
    );
    assert_eq!(program_key(&a), program_key(&b));
    assert_eq!(code_at_frame(&a), code_at_frame(&b), "one compiled region");

    assert_eq!(program_key(&ramp(1.0)), program_key(&ramp(2.5)));
    assert_eq!(code_at_frame(&ramp(1.0)), code_at_frame(&ramp(2.5)));
}

/// A program compiled once, rebound from two different `Args`, gives the
/// pixels of baking each call: the call's values and the `Args` record's are
/// one binding.
#[test]
fn a_program_compiled_once_is_rebound_from_args() {
    let lattice = Lattice::frame(FRAME.0, FRAME.1);
    let program = Manifold::compile(&boxed(Bounds::default(), 0.0, 0.0), lattice.extent);
    let bound = program.bind(&[]);
    let calls = [
        BoxedArgs {
            b: UNIT,
            fg: 1.0,
            bg: 0.25,
        },
        BoxedArgs {
            b: Bounds {
                x0: 0.0,
                y0: 3.0,
                x1: 9.0,
                y1: 6.0,
            },
            fg: -2.0,
            bg: 5.0,
        },
    ];
    let mut block = program.block();
    for args in calls {
        args.write_into(&mut block).expect("boxed's own arguments");
        let rebound = lattice.collapse(&bound.clone().with_uniforms(&block));
        let baked = lattice.bake(&boxed(args.b, args.fg, args.bg));
        assert_eq!(rebound.buffer(), baked.buffer(), "{args:?}");
    }

    let program = Manifold::compile(&ramp(0.0), lattice.extent);
    let mut block = program.block();
    RampArgs { scale: 2.0 }
        .write_into(&mut block)
        .expect("ramp's own argument");
    let rebound = lattice.collapse(&program.bind(&[]).with_uniforms(&block));
    assert_eq!(rebound.buffer(), lattice.bake(&ramp(2.0)).buffer());
}

/// One program for every count (L9): a count is a uniform, keyed by its slot,
/// so two calls with different counts share a canonical key and a compiled
/// region.
#[test]
fn every_count_is_one_program() {
    assert_eq!(program_key(&sum_to(3)), program_key(&sum_to(7)));
    assert_eq!(
        code_at_frame(&sum_to(3)),
        code_at_frame(&sum_to(7)),
        "one compiled region"
    );
}

/// A count is one uniform, declared in parameter order as the `f32` it names,
/// and a program compiled at one count is rebound to another from `Args`: the
/// pixels of baking that call. `2²⁴` itself is a count, and is written.
#[test]
fn a_program_compiled_at_one_count_is_rebound_to_another() {
    let defaults: Vec<f32> = scaled_sum(UNIT, 5, 0.5)
        .uniforms()
        .iter()
        .map(|u| u.default)
        .collect();
    assert_eq!(defaults, [2.0, 1.0, 6.0, 4.0, 5.0, 0.5]);

    let lattice = Lattice::frame(FRAME.0, FRAME.1);
    let program = Manifold::compile(&sum_to(0), lattice.extent);
    let bound = program.bind(&[]);
    let mut block = program.block();
    for n in [7, 0, 3] {
        SumToArgs { n }
            .write_into(&mut block)
            .expect("sum_to's own count");
        let rebound = lattice.collapse(&bound.clone().with_uniforms(&block));
        assert_eq!(
            rebound.buffer(),
            lattice.bake(&sum_to(n)).buffer(),
            "n = {n}"
        );
    }
    let at_the_top = sum_to(16_777_216);
    assert_eq!(at_the_top.uniforms()[0].default, 16_777_216.0);
    SumToArgs { n: 16_777_216 }
        .write_into(&mut block)
        .expect("2^24 is a count");
    assert_eq!(block.values(), [16_777_216.0]);
}

/// The host function refuses a count past 2²⁴, as `Kernel::over` refuses a
/// literal range past it: its uniform would hold `2²⁴`, a count, and the fold
/// would run one trip short with no NaN to say so (§1.4).
#[test]
#[should_panic(expected = "past 16777216 (2^24)")]
fn the_host_function_refuses_a_count_past_the_domain() {
    let _refused = sum_to(16_777_217);
}

/// The `Args` write refuses a count past 2²⁴ before it writes anything (§1.4):
/// as an `f32`, `2²⁴ + 1` is `2²⁴`, a count, so the program could not tell.
/// That nothing is written is the order of the expansion, which refuses before
/// `set_declared` (`emit.rs`'s unit test).
#[test]
#[should_panic(expected = "past 16777216 (2^24)")]
fn the_args_write_refuses_a_count_past_the_domain() {
    let program = Manifold::compile(&sum_to(0), [1, 1]);
    let mut block = program.block();
    let _refused = SumToArgs { n: 16_777_217 }.write_into(&mut block);
}

/// How long a fold outside its domain may take to answer: it never hangs.
const HANG: Duration = Duration::from_secs(60);

/// Run `f` on a thread and wait for it a minute: a fold that hangs is a
/// failed assertion, not a CI timeout. A panic in `f` is the caller's panic,
/// with its message.
fn within<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        if let Err(gone) = tx.send(f()) {
            panic!("the test stopped listening before its kernel returned: {gone}");
        }
    });
    match rx.recv_timeout(HANG) {
        Ok(value) => value,
        Err(RecvTimeoutError::Timeout) => {
            panic!("a count outside the domain never hangs: no answer in {HANG:?}")
        }
        Err(RecvTimeoutError::Disconnected) => match worker.join() {
            Err(panic) => std::panic::resume_unwind(panic),
            Ok(()) => unreachable!("a worker that sent its answer is not disconnected"),
        },
    }
}

/// A block written by position is untyped: any `f32` reaches the count's
/// slot, and one that names no count makes this fold NaN (L5), the loop not
/// entered, so nothing hangs. (`2²⁴ + 1` cannot be written this way at all:
/// no `f32` names it, `value_bounds_jit.rs`'s
/// `the_f32_nearest_two_to_the_24_plus_one_is_two_to_the_24`.)
#[test]
fn a_count_written_raw_outside_the_domain_is_nan() {
    let values = within(|| {
        let program = Manifold::compile(&sum_to(0), [1, 1]);
        let bound = program.bind(&[]);
        let mut block = program.block();
        [16_777_218.0, f32::NAN, -1.0, 2.5, f32::INFINITY, 1e30]
            .into_iter()
            .map(|raw| {
                block.set_declared([raw]).expect("one argument");
                bound.clone().with_uniforms(&block).eval_at(3.0, 0.0)
            })
            .collect::<Vec<f32>>()
    });
    assert!(values.iter().all(|v| v.is_nan()), "{values:?}");
}

/// A parameter the body never reads is still declared, so the arguments
/// after it keep their positions: its value is taken and dropped, not given
/// to the next one.
#[test]
fn an_unread_parameter_binds_positionally() {
    let k = skips(99.0, 1.0, 3.0);
    assert_eq!(k.uniforms().len(), 3, "the unread parameter is declared");
    assert_eq!(Lattice::eval_at(&k, 0.0, 0.0), 13.0);

    let program = Manifold::compile(&k, [1, 1]);
    assert_eq!(program.uniforms().len(), 2, "the code reads `a` and `c`");
    let mut block = program.block();
    SkipsArgs {
        unread: 42.0,
        a: 2.0,
        c: 5.0,
    }
    .write_into(&mut block)
    .expect("skips's own arguments");
    assert_eq!(
        program.bind(&[]).with_uniforms(&block).eval_at(0.0, 0.0),
        25.0,
        "a = 2 and c = 5; a shifted binding would give 42 · 10 + 2"
    );
}

/// `Args` written for a program of another arity is a typed error, as are
/// values of the wrong count and an entry's `Args` for a composition that
/// declares other arguments beside the entry's; none truncates or pads, and
/// nothing is written.
#[test]
fn an_arity_mismatch_is_an_error() {
    let program = Manifold::compile(&boxed(UNIT, 1.0, 0.0), [1, 1]);
    let mut block = program.block();
    let skips_args = SkipsArgs {
        unread: 0.0,
        a: 1.0,
        c: 2.0,
    };
    assert_eq!(
        skips_args.write_into(&mut block),
        Err(ArityMismatch {
            declared: 6,
            supplied: 3
        })
    );
    assert_eq!(
        block.set_declared([1.0; 7]),
        Err(ArityMismatch {
            declared: 6,
            supplied: 7
        })
    );
    assert_eq!(block.values(), program.block().values(), "nothing written");

    let beside = Uniform::new(1.0).kernel().add(&boxed(UNIT, 1.0, 0.0));
    let program = Manifold::compile(&beside, [1, 1]);
    let boxed_args = BoxedArgs {
        b: UNIT,
        fg: 1.0,
        bg: 0.0,
    };
    assert_eq!(
        boxed_args.write_into(&mut program.block()),
        Err(ArityMismatch {
            declared: 7,
            supplied: 6
        }),
        "the uniform's argument is declared first, then boxed's six"
    );
}

/// A kernel the entry's is composed into, beside kernels with no arguments
/// of their own, declares the entry's arguments in the entry's order — as
/// the right operand of an arithmetic node or a warp's coordinate, which a
/// composition splices in rather than copies — so the entry's `Args`
/// rebinds it: the pixels are those of baking the same composition of the
/// call. An unread parameter keeps its place there too.
#[test]
fn the_entrys_args_rebind_a_kernel_it_is_composed_into() {
    let lattice = Lattice::frame(FRAME.0, FRAME.1);
    let compositions: [fn(&Kernel) -> Kernel; 3] = [
        |k| Kernel::constant(0.5).add(k),
        |k| Kernel::x().mul(k),
        |k| Kernel::x().add(&Kernel::y()).at(k, &Kernel::y()),
    ];
    let args = BoxedArgs {
        b: Bounds {
            x0: 1.0,
            y0: 2.0,
            x1: 7.0,
            y1: 5.0,
        },
        fg: 3.0,
        bg: -0.5,
    };
    for (i, compose) in compositions.iter().enumerate() {
        let program = Manifold::compile(&compose(&boxed(UNIT, 1.0, 0.25)), lattice.extent);
        let mut block = program.block();
        args.write_into(&mut block)
            .expect("the entry's arguments, in its order");
        let rebound = lattice.collapse(&program.bind(&[]).with_uniforms(&block));
        let baked = lattice.bake(&compose(&boxed(args.b, args.fg, args.bg)));
        assert_eq!(rebound.buffer(), baked.buffer(), "composition {i}");
    }

    let composed = Kernel::constant(1.0).add(&skips(99.0, 1.0, 3.0));
    assert_eq!(composed.uniforms().len(), 3, "the unread parameter joins");
    let program = Manifold::compile(&composed, [1, 1]);
    let mut block = program.block();
    SkipsArgs {
        unread: 42.0,
        a: 2.0,
        c: 5.0,
    }
    .write_into(&mut block)
    .expect("skips's own arguments");
    assert_eq!(
        program.bind(&[]).with_uniforms(&block).eval_at(0.0, 0.0),
        26.0,
        "1 + a·10 + c with a = 2 and c = 5"
    );
}

/// How far a quotient may sit from its value, relative, at every ISA tier.
///
/// The optimizer may divide through `Recip`, which is an estimate whose
/// accuracy is the tier's (CLAUDE.md, "Floating point at the edges"):
/// `rcpps`, AVX2's, is good to about 1.5·2⁻¹² and gives `3 / 4` as
/// `0.7498169`; AVX-512's `vrcp14ps` happens to be exact on `1/4`; NEON's
/// `FRECPE` plus one `FRECPS` step is closer than `rcpps`. A quotient pinned
/// bit for bit is a pin on one tier. 2⁻¹¹ admits every tier's estimate and
/// is far inside the gap between one ring count and the next.
const RECIP_RELATIVE_TOLERANCE: f32 = 1.0 / 2048.0;

/// The macro doc's binding-times example: one call baked, and the program
/// compiled once and rebound from `Args` with another call's values, which
/// gives that call's pixels — exactly, since both run one program at one
/// tier.
#[test]
fn the_macro_docs_example_bakes_and_rebinds() {
    let lattice = Lattice::frame(FRAME.0, FRAME.1);
    let b = Bounds {
        x0: 1.0,
        y0: 1.0,
        x1: 8.0,
        y1: 5.0,
    };
    let once = lattice.bake(&rings(b, 1.0, 0.0));
    let corner = once.buffer()[0];
    assert!(
        (corner - 0.75).abs() <= 0.75 * RECIP_RELATIVE_TOLERANCE,
        "(0, 0) is inside, √2 from the corner: within three of the four rings, so 3/4 \
         up to the tier's reciprocal; got {corner}"
    );

    let program = Manifold::compile(&rings(b, 1.0, 0.0), lattice.extent);
    let mut block = program.block();
    RingsArgs {
        b,
        fg: 0.5,
        bg: 0.25,
    }
    .write_into(&mut block)
    .expect("rings' own arguments");
    let again = lattice.collapse(&program.bind(&[]).with_uniforms(&block));
    assert_eq!(again.buffer(), lattice.bake(&rings(b, 0.5, 0.25)).buffer());
}
