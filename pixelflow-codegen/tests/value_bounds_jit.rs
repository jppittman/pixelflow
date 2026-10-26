//! A fold whose bound is a value runs, on every tier, and computes its fold or
//! NaN: never a hang, never a trip count converted to an integer, never a read
//! past a buffer (docs/language/2026-10-20-a-bound-is-a-value.md, L2 and L5).
//!
//! For a bound `v` at run time, `v` is a count when `0 <= v <= 2^24` and
//! `floor(v) == v`. Then the fold is `⊕_{k ∈ [lo, hi)}`, the monoid's identity
//! over a range that runs backwards; and for any other `v` — a fraction, a
//! negative, NaN, an infinity, `2^24 + 2` — the fold is NaN. The guard that
//! says so is `passes::legalize`'s last pass, and the oracles here are
//! explicit folds in scalar Rust, from the monoid's identity.
//!
//! Every count is an integer below `2^24` and every sample is an integer, so
//! each sum is exact in any order of summation (the optimizer may reassociate)
//! and the answer is compared bit for bit where it is finite.
//!
//! A kernel runs two ways. `run_raw` compiles the arena as written, through
//! `emit::compile`: the loops are the ones written, and the guard and the
//! allocator are what is under test. `run_linked` goes through
//! `jit_cache::compile`, the optimizer first: it may peel a fold or factor an
//! invariant operand out of it, and the answer must not change.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use std::sync::Arc;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

use pixelflow_codegen::emit::{self, EmitTraffic, Trips};
use pixelflow_codegen::isa::{self, Isa};
use pixelflow_codegen::jit_cache;
use pixelflow_ir::fold::{Binder, Bound, Fold, Monoid};
use pixelflow_ir::{ExprArena, ExprId, Kernel, LatticeShape, OpKind, Uniform};

/// Wide enough for a main column fold at every lane count (4, 8 and 16), and
/// several rows so the row fold is a loop around the columns.
const SHAPE: LatticeShape = LatticeShape::new([64, 4]);

/// A width that leaves a remainder at every lane count, so the column fold is
/// strip-mined and the same `Write` is stored by a main fold and a remainder.
const RAGGED: LatticeShape = LatticeShape::new([37, 3]);

/// Where the lattice's sample `(0, 0)` lies: integers, so every sample is.
const ORIGIN: [f32; 2] = [1.0, 2.0];

/// The most a test waits for a kernel before it calls it a hang.
const HANG: Duration = Duration::from_secs(60);

/// `2^24`, the largest count: the last integer to which an `f32` names every
/// integer.
const TOP: f32 = 16_777_216.0;

/// One value of each kind that names no count: past `2^24`, NaN, a fraction, a
/// negative, both infinities and a huge finite one. No subnormal: whether one
/// reads as 0 depends on the thread's denormal mode, not on the guard.
const OUTSIDE: [f32; 7] = [
    TOP + 2.0,
    f32::NAN,
    2.5,
    -1.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    1e30,
];

/// A per-call count: a uniform holding `v`.
fn count(v: f32) -> Kernel {
    Uniform::new(v).kernel()
}

/// `0` as a bound.
fn zero() -> Kernel {
    Kernel::constant(0.0)
}

/// Whether `v` is a count, the domain a bound's guard admits: an integer in
/// `0..=2^24`.
fn names_a_count(v: f32) -> bool {
    (0.0..=TOP).contains(&v) && v.fract() == 0.0
}

/// Run `f` on a thread and wait for it a minute: a fold that hangs is a failed
/// assertion, not a CI timeout. A panic in `f` is the caller's panic, with its
/// message.
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
            panic!("a fold whose bound is outside its domain never hangs: no answer in {HANG:?}")
        }
        Err(RecvTimeoutError::Disconnected) => match worker.join() {
            Err(panic) => std::panic::resume_unwind(panic),
            Ok(()) => unreachable!("a worker that sent its answer is not disconnected"),
        },
    }
}

/// `arena` compiled as written at `shape`, called with `block` as its uniform
/// block, and the plane it fills with the traffic the compile counted.
fn run_arena(
    arena: &ExprArena,
    root: ExprId,
    shape: LatticeShape,
    block: &[f32],
) -> (Vec<f32>, EmitTraffic) {
    let compiled = emit::compile(arena, root, shape).expect("the kernel compiles");
    (call(&compiled.code, block, shape), compiled.traffic)
}

/// The plane `code` fills at `shape`, its uniform block `block`.
fn call(code: &emit::CompiledKernel, block: &[f32], shape: LatticeShape) -> Vec<f32> {
    let [width, height] = shape.extent().map(|n| n as usize);
    let mut out = vec![f32::NAN; width * height];
    // No buffers: the uniform block, then the origin's.
    let ctx: [*const f32; 2] = [block.as_ptr(), ORIGIN.as_ptr()];
    // SAFETY: the kernels here declare no buffer, `ctx` holds the uniform
    // block they were linked against and the origin block, each live for the
    // call, and `out` is the whole plane of the extent they were compiled at.
    unsafe { code.call(ctx.as_ptr(), out.as_mut_ptr(), width) };
    out
}

/// `kernel` as written, with the uniforms at their defaults.
fn run_raw(kernel: &Kernel, shape: LatticeShape) -> (Vec<f32>, EmitTraffic) {
    let (arena, root) = kernel.parts();
    let block: Vec<f32> = arena.uniforms().iter().map(|d| d.default).collect();
    run_arena(arena, root, shape, &block)
}

/// `kernel` through the optimizer and the shape-keyed cache, with the uniforms
/// at their defaults in the link's order.
fn run_linked(kernel: &Kernel, shape: LatticeShape) -> Vec<f32> {
    let linked = jit_cache::compile(kernel, shape).expect("the kernel compiles");
    let block: Vec<f32> = linked.uniforms.iter().map(|d| d.default).collect();
    call(&linked.kernel, &block, shape)
}

/// What a sample's `(x, y)` is, at `(col, row)`.
fn at(shape: LatticeShape, i: usize) -> (f32, f32) {
    let width = shape.extent()[0] as usize;
    (
        ORIGIN[0] + (i % width) as f32,
        ORIGIN[1] + (i / width) as f32,
    )
}

/// A built kernel's answer on both paths, labelled, each inside the watchdog.
fn both(
    shape: LatticeShape,
    build: impl Fn() -> Kernel + Send + Copy + 'static,
) -> [(&'static str, Vec<f32>); 2] {
    [
        ("as written", within(move || run_raw(&build(), shape).0)),
        ("optimized", within(move || run_linked(&build(), shape))),
    ]
}

/// Every sample of both paths is `want(x, y)`, bit for bit; a NaN is only a
/// NaN.
#[track_caller]
fn assert_plane<const PATHS: usize>(
    shape: LatticeShape,
    planes: [(&'static str, Vec<f32>); PATHS],
    context: &str,
    want: impl Fn(f32, f32) -> f32,
) {
    for (path, plane) in planes {
        assert_eq!(
            plane.len(),
            (shape.extent()[0] * shape.extent()[1]) as usize
        );
        for (i, &got) in plane.iter().enumerate() {
            let (x, y) = at(shape, i);
            let want = want(x, y);
            let same = match want.is_nan() {
                true => got.is_nan(),
                false => got.to_bits() == want.to_bits(),
            };
            assert!(
                same,
                "{context} ({path}) at ({x}, {y}): got {got} ({:#010x}), want {want} ({:#010x})",
                got.to_bits(),
                want.to_bits()
            );
        }
    }
}

/// `Σ_{k<n} (X + k)`, one program for every `n`.
fn sum_of_x_plus_k(n: f32) -> Kernel {
    Kernel::over_values(Monoid::SUM, &zero(), &count(n), |k| Kernel::x().add(k))
}

/// The oracle for [`sum_of_x_plus_k`] and its relatives: an explicit fold from
/// the monoid's identity over `lo..hi`, of `term(k)`.
fn fold_of(monoid: Monoid, lo: u64, hi: u64, term: impl Fn(f32) -> f32) -> f32 {
    let identity = f32::from_bits(monoid.identity().bits());
    let combine = |a: f32, b: f32| match monoid {
        Monoid::SUM => a + b,
        Monoid::PRODUCT => a * b,
        Monoid::MIN => a.min(b),
        Monoid::MAX => a.max(b),
        _ => unreachable!("a value bound takes SUM, PRODUCT, MIN or MAX"),
    };
    (lo..hi).fold(identity, |acc, k| combine(acc, term(k as f32)))
}

/// A count in the domain runs its fold: `n` trips, `n = 0` and `-0.0` the
/// identity.
#[test]
fn a_count_in_the_domain_runs_its_fold() {
    for n in [0.0f32, -0.0, 1.0, 7.0, 1000.0] {
        let planes = both(SHAPE, move || sum_of_x_plus_k(n));
        assert_plane(SHAPE, planes, &format!("Σ_{{k<{n}}} (X + k)"), |x, _| {
            fold_of(Monoid::SUM, 0, n as u64, |k| x + k)
        });
    }
}

/// The column fold's remainder stores the same value as its main fold, and
/// both read the one guard.
#[test]
fn a_ragged_row_runs_the_guard_in_both_column_folds() {
    for n in [0.0f32, 7.0, f32::NAN] {
        let planes = both(RAGGED, move || sum_of_x_plus_k(n));
        assert_plane(
            RAGGED,
            planes,
            &format!("ragged, n = {n}"),
            |x, _| match names_a_count(n) {
                true => fold_of(Monoid::SUM, 0, n as u64, |k| x + k),
                false => f32::NAN,
            },
        );
    }
}

/// `2^24` is the last count: the counter reaches it exactly, and the latch's
/// compare fires there. The body is constant, so the loop runs once per call.
#[test]
fn a_count_at_the_exact_bound_runs_every_trip() {
    let build =
        || Kernel::over_values(Monoid::SUM, &zero(), &count(TOP), |_| Kernel::constant(1.0));
    let planes = both(LatticeShape::POINT, build);
    assert_plane(LatticeShape::POINT, planes, "Σ_{k<2^24} 1", |_, _| TOP);
}

/// Outside the domain the fold is NaN, and the loop is not entered. Unguarded,
/// a `hi` of `+∞`, `1e30` or `2^24 + 2` stalls an `f32` counter at `2^24` for
/// ever, on every tier. A NaN `hi` hangs it too where the latch's compare is
/// ordered (NEON's `FCMGE`); x86's is unordered, so a NaN end exits after one
/// trip there. The watchdog turns a hang into a failure.
#[test]
fn a_count_outside_the_domain_is_nan_and_never_hangs() {
    for n in OUTSIDE {
        let planes = both(SHAPE, move || sum_of_x_plus_k(n));
        assert_plane(SHAPE, planes, &format!("n = {n}"), |_, _| f32::NAN);
    }
}

/// `Σ_{k<n} z·(X + k)` for a factor `z` the optimizer proves zero, spelled as
/// the literal `0` and as `X − X`: `FactorFold` takes `z` out of the fold and
/// the `Mul` annihilator (`0·y → 0`) removes the fold, so the optimizer leaves
/// no fold for the guard to wrap. In the domain both paths give `0`. Outside
/// it the fold as written is NaN, and the optimized kernel is exactly `+0.0`:
/// the annihilation rule's answer, not the guard's, which `Kernel::over_values`
/// documents.
#[test]
fn the_annihilator_erases_the_domain_nan_of_a_zero_factored_fold() {
    /// How the zero factor is spelled.
    type Factor = fn() -> Kernel;
    fn x_minus_x() -> Kernel {
        Kernel::x().sub(&Kernel::x())
    }
    let factors: [(&str, Factor); 2] = [("0", zero), ("X − X", x_minus_x)];
    for (spelled, factor) in factors {
        let zero_times = move |n: f32| {
            Kernel::over_values(Monoid::SUM, &zero(), &count(n), move |k| {
                factor().mul(&Kernel::x().add(k))
            })
        };
        for n in [0.0f32, 3.0] {
            let planes = both(SHAPE, move || zero_times(n));
            assert_plane(
                SHAPE,
                planes,
                &format!("({spelled})·(X + k), n = {n}"),
                |_, _| 0.0,
            );
        }
        for n in [f32::NAN, 2.5] {
            let written = within(move || run_raw(&zero_times(n), SHAPE).0);
            assert_plane(
                SHAPE,
                [("as written", written)],
                &format!("({spelled})·(X + k), n = {n}"),
                |_, _| f32::NAN,
            );
            let optimized = within(move || run_linked(&zero_times(n), SHAPE));
            assert_plane(
                SHAPE,
                [("optimized", optimized)],
                &format!("({spelled})·(X + k), n = {n}"),
                |_, _| 0.0,
            );
        }
    }
}

/// A value `lo` outside the domain beside a literal `hi` is NaN too, and the
/// loop is not entered: the guard tests `lo` although `hi` is a count already.
/// Unguarded, `-∞` hangs on every tier, for `-∞ + 1` never reaches `9`. A NaN
/// `lo` hangs too where the latch's compare is ordered (NEON), and exits after
/// one trip on x86, where it is unordered.
#[test]
fn a_value_lo_outside_the_domain_is_nan_and_never_hangs() {
    for lo in OUTSIDE {
        let planes = both(SHAPE, move || {
            Kernel::over_values(Monoid::SUM, &count(lo), &Kernel::constant(9.0), |k| {
                Kernel::x().add(k)
            })
        });
        assert_plane(SHAPE, planes, &format!("[{lo}, 9)"), |_, _| f32::NAN);
    }
}

/// A pair of two values, one outside the domain and the other in it, is NaN
/// whichever end is outside: each end is tested, not only the nearer one.
#[test]
fn a_pair_with_one_value_outside_the_domain_is_nan() {
    for outside in OUTSIDE {
        let planes = both(SHAPE, move || {
            Kernel::over_values(Monoid::SUM, &count(outside), &count(9.0), |k| {
                Kernel::x().add(k)
            })
        });
        assert_plane(SHAPE, planes, &format!("[{outside}, 9)"), |_, _| f32::NAN);

        let planes = both(SHAPE, move || {
            Kernel::over_values(Monoid::SUM, &count(3.0), &count(outside), |k| {
                Kernel::x().add(k)
            })
        });
        assert_plane(SHAPE, planes, &format!("[3, {outside})"), |_, _| f32::NAN);
    }
}

/// The four monoids a value bound takes, each with its identity.
const MONOIDS: [(Monoid, f32); 4] = [
    (Monoid::SUM, 0.0),
    (Monoid::PRODUCT, 1.0),
    (Monoid::MIN, f32::INFINITY),
    (Monoid::MAX, f32::NEG_INFINITY),
];

/// A value `lo` in the domain beside a literal `hi` runs its fold from there:
/// `Σ_{k ∈ [lo, 1003)} (X + k)` for `lo` at 0, 1 and 7 (1000 is
/// [`a_value_lo_runs_its_fold_up_to_the_top_of_the_domain`]'s).
#[test]
fn a_value_lo_in_the_domain_beside_a_literal_hi_runs_its_fold() {
    const HI: u64 = 1003;
    for lo in [0.0f32, 1.0, 7.0] {
        let planes = both(SHAPE, move || {
            Kernel::over_values(Monoid::SUM, &count(lo), &Kernel::constant(HI as f32), |k| {
                Kernel::x().add(k)
            })
        });
        assert_plane(SHAPE, planes, &format!("[{lo}, {HI})"), |x, _| {
            fold_of(Monoid::SUM, lo as u64, HI, |k| x + k)
        });
    }
}

/// How a bound is spelled: a per-call count (a uniform) or a constant.
type Spelling = fn(f32) -> Kernel;

/// Both spellings, labelled. A count is the same count either way.
const SPELLINGS: [(&str, Spelling); 2] = [("uniform", count), ("constant", Kernel::constant)];

/// A value `lo` runs its fold up to the top of the domain, beside a `hi` that
/// is a per-call count and beside one that is a constant: three trips from
/// 1000 to 1003; two trips from `2^24 - 2` to `2^24`, where the counter
/// reaches `2^24` exactly and the latch fires there, so `MAX` of the index is
/// `2^24 - 1`; and `[2^24, 2^24)`, a count with the range empty, which is each
/// monoid's identity.
#[test]
fn a_value_lo_runs_its_fold_up_to_the_top_of_the_domain() {
    for (spelled, hi_of) in SPELLINGS {
        let planes = both(SHAPE, move || {
            Kernel::over_values(Monoid::SUM, &count(1000.0), &hi_of(1003.0), |k| {
                Kernel::x().add(k)
            })
        });
        assert_plane(SHAPE, planes, &format!("[1000, {spelled} 1003)"), |x, _| {
            fold_of(Monoid::SUM, 1000, 1003, |k| x + k)
        });

        let planes = both(SHAPE, move || {
            Kernel::over_values(Monoid::MAX, &count(TOP - 2.0), &hi_of(TOP), |k| k.clone())
        });
        assert_plane(
            SHAPE,
            planes,
            &format!("MAX_{{k ∈ [2^24 - 2, {spelled} 2^24)}} k"),
            |_, _| TOP - 1.0,
        );

        for (monoid, identity) in MONOIDS {
            let planes = both(SHAPE, move || {
                Kernel::over_values(monoid, &count(TOP), &hi_of(TOP), |k| Kernel::x().add(k))
            });
            assert_plane(
                SHAPE,
                planes,
                &format!("{monoid:?} [2^24, {spelled} 2^24)"),
                |_, _| identity,
            );
        }
    }
}

/// One uniform at both ends is the empty range where it is a count, and NaN
/// where it is none: the range is not empty by being `[v, v)`, so the guard
/// tests `v`, at three counts and at every value of [`OUTSIDE`].
#[test]
fn one_value_at_both_ends_is_the_identity_in_the_domain_and_nan_outside() {
    for v in [0.0f32, 7.0, TOP].into_iter().chain(OUTSIDE) {
        let planes = both(SHAPE, move || {
            let n = Uniform::new(v);
            Kernel::over_values(Monoid::SUM, &n.kernel(), &n.kernel(), |k| {
                Kernel::x().add(k)
            })
        });
        assert_plane(
            SHAPE,
            planes,
            &format!("[{v}, {v})"),
            |_, _| match names_a_count(v) {
                true => 0.0,
                false => f32::NAN,
            },
        );
    }
}

/// `2^24 + 1` is not a case: no `f32` names it, and `16777217.0` rounds to
/// `2^24`, which is a count.
#[test]
fn the_f32_nearest_two_to_the_24_plus_one_is_two_to_the_24() {
    assert_eq!(16_777_217.0f32, TOP);
}

/// A bound a rule or a hand made from a real that names no count is kept a
/// value, not refused (the arena's `push_reduce`), and its guard makes the
/// fold NaN rather than a panic.
#[test]
fn a_literal_a_rule_made_outside_the_domain_is_nan() {
    let mut a = ExprArena::new();
    let binder = Binder::from_slot(0).expect("slot 0 exists");
    let (x, k) = (a.push_var(0), a.push_var(binder.var()));
    let body = a.push_binary(OpKind::Add, x, k);
    let hi = a.push_const(2.5);
    let root = a.push_reduce(
        Fold::new(Monoid::SUM, binder),
        Bound::Lit(0),
        Bound::Value(hi),
        body,
    );
    let (plane, _) = within(move || run_arena(&a, root, SHAPE, &[]));
    assert!(plane.iter().all(|v| v.is_nan()), "{plane:?}");
}

/// A pair that runs backwards at run time is the monoid's identity, for each
/// monoid a value bound takes. The pairs put 0 and 1000 at `hi` and 1 at `lo`,
/// so each end's domain test is read at the small counts and above them.
#[test]
fn a_pair_that_runs_backwards_is_the_identity() {
    for (monoid, identity) in MONOIDS {
        for (lo, hi) in [(5.0f32, 3.0f32), (4.0, 4.0), (1.0, 0.0), (1003.0, 1000.0)] {
            let planes = both(SHAPE, move || {
                Kernel::over_values(monoid, &count(lo), &count(hi), |k| Kernel::x().add(k))
            });
            assert_plane(
                SHAPE,
                planes,
                &format!("{monoid:?} [{lo}, {hi})"),
                |_, _| identity,
            );
        }
    }
}

/// A value `lo` starts the fold there: the counter is seeded with its lane (plus
/// zero, which only the sign of a zero count shows).
#[test]
fn a_value_lo_starts_the_fold() {
    let planes = both(SHAPE, || {
        Kernel::over_values(Monoid::SUM, &count(3.0), &count(7.0), |k| {
            Kernel::x().add(k)
        })
    });
    assert_plane(SHAPE, planes, "Σ_{k=3}^{6} (X + k)", |x, _| {
        fold_of(Monoid::SUM, 3, 7, |k| x + k)
    });
}

/// `-0.0` is the count 0 at `lo`, and the fold starts at the integer 0, `+0.0`,
/// whether the count is a uniform or a constant (`Bound::discover` turns a
/// literal `-0.0` into `Lit(0)`). A body that sees the sign of zero tells them
/// apart: `1/k` is `+∞` at `+0.0` and `-∞` at `-0.0`, and a `MAX` over one trip
/// keeps the sign of the index it was given.
#[test]
fn a_value_lo_of_negative_zero_starts_the_fold_at_positive_zero() {
    for (kind, lo_of) in SPELLINGS {
        for lo in [-0.0f32, 0.0] {
            let planes = both(SHAPE, move || {
                Kernel::over_values(Monoid::MIN, &lo_of(lo), &count(1.0), |k| {
                    Kernel::constant(1.0).div(k)
                })
            });
            assert_plane(
                SHAPE,
                planes,
                &format!("MIN_{{k ∈ [{kind} {lo:?}, 1)}} 1/k"),
                |_, _| f32::INFINITY,
            );

            let planes = both(SHAPE, move || {
                Kernel::over_values(Monoid::MAX, &lo_of(lo), &count(1.0), |k| k.clone())
            });
            assert_plane(
                SHAPE,
                planes,
                &format!("MAX_{{k ∈ [{kind} {lo:?}, 1)}} k"),
                |_, _| 0.0,
            );
        }
    }
}

/// `Σ_{i<8} Σ_{j<i} (X + j)`: the inner `hi` is the outer fold's index, so
/// the guard runs once per outer trip. As written the inner bound stays a
/// value; optimized, the outer fold may peel and make it known (L8).
#[test]
fn a_triangular_sum_runs_each_inner_fold_to_its_outer_index() {
    let build = || {
        Kernel::sum_over(8, |i| {
            Kernel::over_values(Monoid::SUM, &zero(), i, |j| Kernel::x().add(j))
        })
    };
    let planes = both(SHAPE, build);
    assert_plane(SHAPE, planes, "Σ_i Σ_{j<i} (X + j)", |x, _| {
        fold_of(Monoid::SUM, 0, 8, |i| {
            fold_of(Monoid::SUM, 0, i as u64, |j| x + j)
        })
    });
}

/// The same under a per-call count for the outer bound: `n = 0` and `5`, and
/// NaN, `+∞` and `2^24 + 2`, which are none.
#[test]
fn a_triangular_sum_under_a_count() {
    for n in [0.0f32, 5.0, f32::NAN, f32::INFINITY, TOP + 2.0] {
        let planes = both(SHAPE, move || {
            Kernel::over_values(Monoid::SUM, &zero(), &count(n), |i| {
                Kernel::over_values(Monoid::SUM, &zero(), i, |j| Kernel::x().add(j))
            })
        });
        assert_plane(
            SHAPE,
            planes,
            &format!("outer n = {n}"),
            |x, _| match names_a_count(n) {
                true => fold_of(Monoid::SUM, 0, n as u64, |i| {
                    fold_of(Monoid::SUM, 0, i as u64, |j| x + j)
                }),
                false => f32::NAN,
            },
        );
    }
}

/// Two folds each over a count of its own, the inner one inside the outer's
/// body: the inner guard is the outer loop's, run once per outer trip, and
/// either count outside the domain makes the whole NaN.
#[test]
fn two_counts_nest() {
    let (nan, inf, past) = (f32::NAN, f32::INFINITY, TOP + 2.0);
    for (n, m) in [
        (0.0f32, 3.0f32),
        (3.0, 0.0),
        (3.0, 4.0),
        (nan, 4.0),
        (3.0, nan),
        (inf, 4.0),
        (3.0, inf),
        (past, 4.0),
        (3.0, past),
    ] {
        let planes = both(SHAPE, move || {
            Kernel::over_values(Monoid::SUM, &zero(), &count(n), |i| {
                Kernel::over_values(Monoid::SUM, &zero(), &count(m), |j| {
                    Kernel::x().add(i).add(j)
                })
            })
        });
        assert_plane(SHAPE, planes, &format!("n = {n}, m = {m}"), |x, _| {
            match names_a_count(n) && names_a_count(m) {
                true => fold_of(Monoid::SUM, 0, n as u64, |i| {
                    fold_of(Monoid::SUM, 0, m as u64, |j| x + i + j)
                }),
                false => f32::NAN,
            }
        });
    }
}

/// A bound the body reads as well: `Σ_{k<n} k·n`, one uniform, two readers.
#[test]
fn a_bound_the_body_reads_too() {
    for n in [0.0f32, 1.0, 7.0, 2.5] {
        let planes = both(SHAPE, move || {
            let n = Uniform::new(n);
            Kernel::over_values(Monoid::SUM, &zero(), &n.kernel(), |k| k.mul(&n.kernel()))
        });
        assert_plane(
            SHAPE,
            planes,
            &format!("n = {n}"),
            |_, _| match names_a_count(n) {
                true => fold_of(Monoid::SUM, 0, n as u64, |k| k * n),
                false => f32::NAN,
            },
        );
    }
}

/// A literal `lo` beside a value `hi`: only the value is in the domain test.
#[test]
fn a_literal_lo_beside_a_value_hi() {
    for n in [0.0f32, 2.0, 3.0, 9.0, -1.0] {
        let planes = both(SHAPE, move || {
            Kernel::over_values(Monoid::SUM, &Kernel::constant(2.0), &count(n), |k| {
                Kernel::x().add(k)
            })
        });
        assert_plane(
            SHAPE,
            planes,
            &format!("[2, {n})"),
            |x, _| match names_a_count(n) {
                true => fold_of(Monoid::SUM, 2, n as u64, |k| x + k),
                false => f32::NAN,
            },
        );
    }
}

/// `Σ_{i<8} Σ_{j∈[i,8)} (X + j)`: a value `lo` that is an enclosing index.
/// The inner counter is seeded with `i + 0.0`, a fresh value (a value `lo` is
/// always seeded its lane plus zero), so the inner loop's entry argument is
/// not the outer counter itself, which the outer latch steps. Each trip of
/// the outer fold starts the inner one at that trip's index.
#[test]
fn an_inner_fold_starting_at_the_outer_index() {
    let build = || {
        Kernel::sum_over(8, |i| {
            Kernel::over_values(Monoid::SUM, i, &Kernel::constant(8.0), |j| {
                Kernel::x().add(j)
            })
        })
    };
    let planes = both(SHAPE, build);
    assert_plane(SHAPE, planes, "Σ_i Σ_{j∈[i,8)} (X + j)", |x, _| {
        fold_of(Monoid::SUM, 0, 8, |i| {
            fold_of(Monoid::SUM, i as u64, 8, |j| x + j)
        })
    });
}

/// `X + Σ_{k<n} k`: the fold reads no coordinate, so its guard and its loop
/// are call-invariant and the carve gives them the body, under the guard's
/// arms. The loop runs once per call, not once per sample.
#[test]
fn a_fold_that_reads_no_coordinate_is_guarded_once_per_call() {
    for n in [0.0f32, 7.0, f32::NAN, f32::INFINITY, TOP + 2.0] {
        let (plane, traffic) = within(move || {
            let sum = Kernel::over_values(Monoid::SUM, &zero(), &count(n), |k| k.clone());
            run_raw(&Kernel::x().add(&sum), SHAPE)
        });
        let want = |x: f32| match names_a_count(n) {
            true => x + fold_of(Monoid::SUM, 0, n as u64, |k| k),
            false => f32::NAN,
        };
        assert_plane(
            SHAPE,
            [("as written", plane)],
            &format!("n = {n}"),
            |x, _| want(x),
        );

        let estimates: Vec<_> = traffic
            .trips
            .iter()
            .filter(|t| matches!(t, Trips::Estimated(_)))
            .collect();
        assert_eq!(
            estimates,
            [&Trips::Estimated(Fold::ESTIMATED_TRIPS)],
            "the one loop with a bound that is a value runs once per call: {:?}",
            traffic.trips
        );
        assert_eq!(traffic.dynamic_memory_ops(), None);
    }
}

/// More values live at once than any tier's vector file holds.
const WALL: u32 = 48;

/// `Σ_m (k + X)·(m + 1)` summed in both orders: [`WALL`] products all live
/// until the last is added, so every vector register is reused inside a loop
/// whose body this is.
fn a_wall_of_values(k: &Kernel) -> Kernel {
    let terms: Vec<Kernel> = (0..WALL)
        .map(|m| Kernel::x().add(k).mul(&Kernel::constant((m + 1) as f32)))
        .collect();
    let chain = |terms: &[Kernel]| {
        terms[1..]
            .iter()
            .fold(terms[0].clone(), |acc, term| acc.add(term))
    };
    let reversed: Vec<Kernel> = terms.iter().rev().cloned().collect();
    chain(&terms).add(&chain(&reversed))
}

/// A wall of values live at once, so `hi`, which the body does not read,
/// survives to the latch only if its interval spans the loop.
///
/// `Σ_{k<n} Σ_m (X + k)·(m + 1)`, twice over.
#[test]
fn a_bound_read_only_by_the_loop_reaches_its_latch() {
    for n in [0.0f32, 1.0, 7.0, f32::NAN] {
        let (plane, _) = within(move || {
            run_raw(
                &Kernel::over_values(Monoid::SUM, &zero(), &count(n), a_wall_of_values),
                SHAPE,
            )
        });
        let weight: f32 = (1..=WALL).map(|m| m as f32).sum();
        assert_plane(
            SHAPE,
            [("as written", plane)],
            &format!("n = {n}"),
            |x, _| match names_a_count(n) {
                true => fold_of(Monoid::SUM, 0, n as u64, |k| 2.0 * weight * (x + k)),
                false => f32::NAN,
            },
        );
    }
}

/// The depth at which a nest of per-call counts, each priced at
/// [`Fold::ESTIMATED_TRIPS`], is priced past `u64`.
const PAST_U64: usize = (u64::BITS / Fold::ESTIMATED_TRIPS.ilog2()) as usize + 1;

/// `depth` folds, each `Σ_{i<n}` over one per-call count, around `X + Σ i`.
fn nested_counts(depth: usize, n: &Uniform, acc: Kernel) -> Kernel {
    if depth == 0 {
        return Kernel::x().add(&acc);
    }
    let again = *n;
    Kernel::over_values(Monoid::SUM, &zero(), &n.kernel(), move |i| {
        nested_counts(depth - 1, &again, acc.add(i))
    })
}

/// A frame, wide and tall enough that a nest a few levels shallower than
/// [`PAST_U64`] is priced past `u64` once the row and column folds are in.
const FRAME: LatticeShape = LatticeShape::new([1920, 1080]);

/// A nest priced past `u64` compiles: the allocator weighs a read by the nest's
/// price and adds the weights of a value's reads, so the sum saturates where it
/// would otherwise overflow. Every count is `1`, so the nest runs each body
/// once and the answer is `X`.
#[test]
fn a_nest_priced_past_u64_still_compiles() {
    let shapes_and_depths = [
        (LatticeShape::POINT, PAST_U64 - 1),
        (LatticeShape::POINT, PAST_U64),
        (LatticeShape::POINT, PAST_U64 + 1),
        (FRAME, PAST_U64 - 2),
    ];
    for (shape, depth) in shapes_and_depths {
        let planes = both(shape, move || {
            nested_counts(depth, &Uniform::new(1.0), Kernel::constant(0.0))
        });
        assert_plane(
            shape,
            planes,
            &format!("{depth} nested counts of 1 at {:?}", shape.extent()),
            |x, _| x,
        );
    }
}

/// What the compile counted for `kernel` as written at [`SHAPE`], and the
/// frame it needed.
fn traffic_of(kernel: &Kernel) -> (EmitTraffic, u64) {
    let (arena, root) = kernel.parts();
    let compiled = emit::compile(arena, root, SHAPE).expect("the kernel compiles");
    (compiled.traffic, compiled.frame_bytes)
}

/// How many lanes one batch of the column fold holds on this host.
fn lanes() -> u64 {
    u64::try_from(pixelflow_codegen::jit_vector_bytes() / size_of::<f32>())
        .expect("a lane count fits u64")
}

/// How many times the column fold's body runs per call at [`SHAPE`]: once per
/// batch of every row. [`SHAPE`]'s width divides by every lane count, so no
/// remainder fold runs beside it.
fn column_trips() -> u64 {
    let [width, rows] = SHAPE.extent().map(u64::from);
    rows * width.div_ceil(lanes())
}

/// The scope of `traffic` whose body runs `trips` times per call. A value
/// bound and its literal twin have one nest, so a scope found in the twin,
/// where every count is known, names the same fold in both. A scope is found by
/// how often it runs and not by being the most run: which fold runs most
/// depends on the lane count.
#[track_caller]
fn scope_running(traffic: &EmitTraffic, trips: u64) -> usize {
    let mut running =
        (0..traffic.trips.len()).filter(|&at| traffic.trips[at] == Trips::Known(trips));
    match (running.next(), running.next()) {
        (Some(scope), None) => scope,
        _ => panic!(
            "exactly one scope runs {trips} times per call: {:?}",
            traffic.trips
        ),
    }
}

/// The wall over `hi` trips when `hi` is a count, and over a literal seven
/// when it is `None`: the kernel a value bound is compared with.
fn the_wall_over(hi: Option<f32>) -> Kernel {
    match hi {
        Some(n) => Kernel::over_values(Monoid::SUM, &zero(), &count(n), a_wall_of_values),
        None => Kernel::sum_over(7, a_wall_of_values),
    }
}

/// A value `hi` that the allocator does not keep in a register across the
/// loop is reloaded at every latch: the latch's compare reads a register, so
/// it is one load in the fold's scope per trip. A literal `hi` is a constant
/// the allocator places again at its read, and costs none.
#[test]
fn a_bound_that_is_not_carried_is_reloaded_at_every_latch() {
    let (value, _) = traffic_of(&the_wall_over(Some(7.0)));
    let (literal, _) = traffic_of(&the_wall_over(None));
    let fold = |t: &EmitTraffic| *t.scopes.last().expect("a nest has a body");
    assert_eq!(
        fold(&value).loads,
        fold(&literal).loads + 1,
        "the fold's scope reloads its end once more than a literal's"
    );
    assert_eq!(
        value.trips.last().map(|t| t.known()),
        Some(None),
        "the loop's trips are an estimate: {:?}",
        value.trips
    );
}

/// A guard whose masks are the call's but whose fold is the column's: its two
/// masks are computed once, live across every batch, and are read again by
/// the guard's two `If`s in each, on every tier. Where they live between
/// batches is the tier's: the opmask file (AVX-512) has no register to keep
/// them in, so each batch reloads them and the frame grows by the slots they
/// live in, narrow ones among them; NEON's vector file keeps them resident.
#[test]
fn a_guard_computed_once_per_call_is_read_again_in_every_batch() {
    let over = |hi: Option<f32>| match hi {
        Some(n) => Kernel::x().mul(&sum_of_x_plus_k(n)),
        None => Kernel::x().mul(&Kernel::sum_over(7, |k| Kernel::x().add(k))),
    };
    let (value, value_frame) = traffic_of(&over(Some(7.0)));
    let (literal, literal_frame) = traffic_of(&over(None));
    let column = |t: &EmitTraffic| t.scopes[2];
    assert_eq!(
        column(&value).instructions,
        column(&literal).instructions + 2,
        "the column fold's every batch runs the guard's two selects, one per mask: \
         {:?} against {:?}",
        column(&value),
        column(&literal)
    );
    if isa::detect() != Isa::Avx512 {
        return;
    }
    assert!(
        column(&value).loads > column(&literal).loads,
        "each batch reads the opmasks back: {:?} against {:?}",
        column(&value),
        column(&literal)
    );
    assert!(
        value_frame > literal_frame,
        "the opmasks are slots in the frame: {value_frame} against {literal_frame}"
    );
}

/// `Y·Σ_{k<n} (Y + k)`: a fold that reads `Y` alone, so it sits in the row fold.
fn y_times_a_sum_over_y(n: f32) -> Kernel {
    Kernel::y().mul(&Kernel::over_values(Monoid::SUM, &zero(), &count(n), |k| {
        Kernel::y().add(k)
    }))
}

/// The same guard around a fold that reads `Y` alone sits in the row fold: its
/// two selects run once per row, and the column fold, which runs in every
/// batch, gains nothing. A guard goes where its fold goes, not to the column
/// fold by default.
#[test]
fn a_guard_around_a_fold_that_reads_only_y_is_the_row_folds() {
    let over = |hi: Option<f32>| match hi {
        Some(n) => y_times_a_sum_over_y(n),
        None => Kernel::y().mul(&Kernel::sum_over(7, |k| Kernel::y().add(k))),
    };
    let (value, _) = traffic_of(&over(Some(7.0)));
    let (literal, _) = traffic_of(&over(None));

    // The row fold runs once per row and the column fold once per batch of
    // each. The fold over `k` runs seven times in every row, so a wide enough
    // batch makes it, and not the column fold, the nest's most-run scope.
    let rows = u64::from(SHAPE.extent()[1]);
    let row = scope_running(&literal, rows);
    let column = scope_running(&literal, column_trips());
    assert_eq!(value.trips[row], Trips::Known(rows), "{:?}", value.trips);

    assert_eq!(
        value.scopes[row].instructions,
        literal.scopes[row].instructions + 2,
        "the row fold runs the guard's two selects, one per mask: {:?} against {:?}",
        value.scopes[row],
        literal.scopes[row]
    );
    assert_eq!(
        value.scopes[column].instructions, literal.scopes[column].instructions,
        "the column fold, run in every batch, runs none of the guard: {:?} against {:?}",
        value.scopes[column], literal.scopes[column]
    );
}

/// The row fold's guard computes the fold in the domain and is NaN outside
/// it, and does not hang: the guard sits in the row fold's scope, once per
/// row, and the arm tests are read there.
#[test]
fn a_guard_in_the_row_fold_computes_its_fold_or_nan() {
    for n in [0.0f32, 7.0, f32::NAN, f32::INFINITY, TOP + 2.0] {
        let planes = both(SHAPE, move || y_times_a_sum_over_y(n));
        assert_plane(
            SHAPE,
            planes,
            &format!("Y·Σ_{{k<{n}}} (Y + k)"),
            |_, y| match names_a_count(n) {
                true => y * fold_of(Monoid::SUM, 0, n as u64, |k| y + k),
                false => f32::NAN,
            },
        );
    }
}

/// A guard around a fold that reads an enclosing fold's index sits in that
/// fold's scope, once per trip of it, although the fold also reads `X`:
/// `Σ_{i<8} Σ_{j<8} (X + j + i)` with the inner count a value, against its
/// literal twin. The enclosing fold's scope gains the guard's two selects, and
/// where the masks cannot stay in registers across its trips (AVX-512's
/// opmasks), the reloads; the column fold, which runs in every batch, gains
/// nothing.
#[test]
fn a_guard_around_a_fold_that_reads_an_enclosing_index_is_that_folds() {
    const OUTER: u32 = 8;
    let over = |inner: Option<f32>| {
        Kernel::sum_over(OUTER, |i| match inner {
            Some(n) => Kernel::over_values(Monoid::SUM, &zero(), &count(n), |j| {
                Kernel::x().add(j).add(i)
            }),
            None => Kernel::sum_over(8, |j| Kernel::x().add(j).add(i)),
        })
    };
    let (value, _) = traffic_of(&over(Some(8.0)));
    let (literal, _) = traffic_of(&over(None));

    let column = scope_running(&literal, column_trips());
    let outer = scope_running(&literal, column_trips() * u64::from(OUTER));
    assert_eq!(
        value.trips[outer], literal.trips[outer],
        "the enclosing fold is the same loop: {:?} against {:?}",
        value.trips, literal.trips
    );

    assert_eq!(
        value.scopes[outer].instructions,
        literal.scopes[outer].instructions + 2,
        "the enclosing fold runs the guard's two selects, one per mask, in every trip: \
         {:?} against {:?}",
        value.scopes[outer],
        literal.scopes[outer]
    );
    assert_eq!(
        value.scopes[column].instructions, literal.scopes[column].instructions,
        "the column fold, run in every batch, runs none of the guard: {:?} against {:?}",
        value.scopes[column], literal.scopes[column]
    );
    if isa::detect() != Isa::Avx512 {
        return;
    }
    assert!(
        value.scopes[outer].loads > literal.scopes[outer].loads,
        "each trip reads the opmasks back: {:?} against {:?}",
        value.scopes[outer],
        literal.scopes[outer]
    );
}

/// One program and one code pointer for every `n`: a fresh uniform per count,
/// the cache keyed on structure, and each kernel computing its own `n`.
#[test]
fn every_count_is_one_program() {
    let linked: Vec<_> = [0.0f32, 1.0, 7.0, 1000.0]
        .into_iter()
        .map(|n| {
            let linked = within(move || {
                jit_cache::compile(&sum_of_x_plus_k(n), SHAPE).expect("the kernel compiles")
            });
            (n, linked)
        })
        .collect();
    let (_, first) = &linked[0];
    for (n, each) in &linked {
        assert!(
            Arc::ptr_eq(&first.kernel, &each.kernel),
            "n = {n} compiled to a program of its own"
        );
        let block: Vec<f32> = each.uniforms.iter().map(|d| d.default).collect();
        let code = Arc::clone(&each.kernel);
        let plane = within(move || call(&code, &block, SHAPE));
        let planes = [("shared code", plane)];
        assert_plane(SHAPE, planes, &format!("n = {n}"), |x, _| {
            fold_of(Monoid::SUM, 0, *n as u64, |k| x + k)
        });
    }
}

/// The other three monoids a value bound takes, under the guard. At `n = 0`
/// the fold is the monoid's identity, `−∞` for `MAX` and `+∞` for `MIN`, not
/// the formula that holds for `n ≥ 1`.
#[test]
fn each_monoid_runs_under_its_guard() {
    for n in [0.0f32, 1.0, 7.0, 2.5] {
        let outside = !names_a_count(n);
        let trips = n as u64;

        let planes = both(SHAPE, move || {
            Kernel::over_values(Monoid::MAX, &zero(), &count(n), |k| Kernel::x().sub(k))
        });
        assert_plane(SHAPE, planes, &format!("MAX, n = {n}"), |x, _| {
            match (outside, n == 0.0) {
                (true, _) => f32::NAN,
                (false, true) => f32::NEG_INFINITY,
                (false, false) => x,
            }
        });

        let planes = both(SHAPE, move || {
            Kernel::over_values(Monoid::MIN, &zero(), &count(n), |k| Kernel::x().sub(k))
        });
        assert_plane(SHAPE, planes, &format!("MIN, n = {n}"), |x, _| {
            match (outside, n == 0.0) {
                (true, _) => f32::NAN,
                (false, true) => f32::INFINITY,
                (false, false) => x - (n - 1.0),
            }
        });

        let planes = both(SHAPE, move || {
            Kernel::over_values(Monoid::PRODUCT, &zero(), &count(n), |k| {
                k.add(&Kernel::constant(1.0))
            })
        });
        assert_plane(
            SHAPE,
            planes,
            &format!("PRODUCT, n = {n}"),
            |_, _| match outside {
                true => f32::NAN,
                false => fold_of(Monoid::PRODUCT, 0, trips, |k| k + 1.0),
            },
        );
    }
}

/// A bound that is a mask and a number is a count: `3` where the mask holds
/// and `0` where it does not. `ScheduledOp::is_mask` calls it a number, so it
/// is not refused, and it runs on every tier, at `lo` and at `hi`.
#[test]
fn a_mask_and_a_number_is_a_count_on_every_tier() {
    for (u, trips) in [(5.0f32, 3u64), (-1.0, 0)] {
        let mut a = ExprArena::new();
        let slot = a.declare_uniform(Uniform::new(u).decl());
        let uniform = a.push_uniform(slot);
        let (zero, three) = (a.push_const(0.0), a.push_const(3.0));
        let positive = a.push_binary(OpKind::Lt, zero, uniform);
        let number = a.push_binary(OpKind::BitAnd, positive, three);
        let binder = Binder::from_slot(0).expect("slot 0 exists");
        let (x, k) = (a.push_var(0), a.push_var(binder.var()));
        let body = a.push_binary(OpKind::Add, x, k);

        let ends = a.push_reduce(
            Fold::new(Monoid::SUM, binder),
            Bound::Lit(0),
            Bound::Value(number),
            body,
        );
        let starts = a.push_reduce(
            Fold::new(Monoid::SUM, binder),
            Bound::Value(number),
            Bound::Lit(3),
            body,
        );
        let root = a.push_binary(OpKind::Add, ends, starts);
        let (plane, _) = within(move || run_arena(&a, root, SHAPE, &[u]));
        let planes = [("as written", plane)];
        assert_plane(SHAPE, planes, &format!("u = {u}"), |x, _| {
            fold_of(Monoid::SUM, 0, trips, |k| x + k) + fold_of(Monoid::SUM, trips, 3, |k| x + k)
        });
    }
}

/// `Σ_{k ∈ [0 < u, 3)} (X + k)`, written by hand: the low bound is the mask
/// `0 < u`, which `Kernel::over_values` takes no spelling of.
fn a_fold_that_starts_at_a_mask() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let slot = a.declare_uniform(Uniform::new(5.0).decl());
    let uniform = a.push_uniform(slot);
    let zero = a.push_const(0.0);
    let positive = a.push_binary(OpKind::Lt, zero, uniform);
    let binder = Binder::from_slot(0).expect("slot 0 exists");
    let (x, k) = (a.push_var(0), a.push_var(binder.var()));
    let body = a.push_binary(OpKind::Add, x, k);
    let root = a.push_reduce(
        Fold::new(Monoid::SUM, binder),
        Bound::Value(positive),
        Bound::Lit(3),
        body,
    );
    (a, root)
}

/// A bound the emitter holds as a mask is refused by name, at either end and
/// on every tier alike: a mask names no count. This one is a `lo`.
#[test]
#[should_panic(expected = "BoundSort")]
fn a_bound_that_is_a_mask_is_refused_by_name() {
    let (a, root) = a_fold_that_starts_at_a_mask();
    let _refused = emit::compile(&a, root, SHAPE);
}

/// The cache's entry reaches the same refusal: `Kernel::over_values` takes no
/// computed bound, but `Kernel::from_parts` adopts any arena.
#[test]
#[should_panic(expected = "BoundSort")]
fn a_mask_bound_adopted_whole_is_refused_by_name_through_the_cache() {
    let (a, root) = a_fold_that_starts_at_a_mask();
    let _refused = jit_cache::compile(&Kernel::from_parts(a, root), SHAPE);
}

/// A select between two masks is a mask in value, but no emitter holds it as
/// one, so it is not refused: until a bound is typed, its bits are read as a
/// float on every tier alike. All ones is a NaN, which makes the fold NaN, and
/// all zeros is the count `0`, the identity.
#[test]
fn a_select_between_masks_is_read_as_a_float_on_every_tier() {
    for (u, want) in [(5.0f32, f32::NAN), (-1.0, 0.0)] {
        let mut a = ExprArena::new();
        let slot = a.declare_uniform(Uniform::new(u).decl());
        let uniform = a.push_uniform(slot);
        let (zero, one) = (a.push_const(0.0), a.push_const(1.0));
        let positive = a.push_binary(OpKind::Lt, zero, uniform);
        let above_one = a.push_binary(OpKind::Lt, one, uniform);
        let hi = a.push_ternary(OpKind::If, positive, positive, above_one);
        let binder = Binder::from_slot(0).expect("slot 0 exists");
        let (x, k) = (a.push_var(0), a.push_var(binder.var()));
        let body = a.push_binary(OpKind::Add, x, k);
        let root = a.push_reduce(
            Fold::new(Monoid::SUM, binder),
            Bound::Lit(0),
            Bound::Value(hi),
            body,
        );
        let (plane, _) = within(move || run_arena(&a, root, SHAPE, &[u]));
        assert_plane(
            SHAPE,
            [("as written", plane)],
            &format!("u = {u}"),
            |_, _| want,
        );
    }
}
