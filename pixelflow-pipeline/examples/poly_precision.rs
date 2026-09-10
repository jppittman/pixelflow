//! Accuracy per nanosecond: what does one more polynomial term actually buy?
//!
//! `horner_vs_estrin` compares the two schedules at fixed degree. That is the
//! wrong axis for the question a shorter critical path is *supposed* to answer:
//! if extra terms are cheap, buy more of them and spend the discount on
//! accuracy. This measures that directly — fit `exp2` on `[0, 1]` (the function
//! [`EXP2_POLY`] approximates) at rising degree, and report the error the JIT
//! actually delivers against the cost it actually pays.
//!
//! The answer is a property of `f32`, not of either schedule: approximation
//! error falls ~30× per added term until it reaches the format's rounding
//! floor, and then stops, flat, while cost keeps rising. Past the knee there is
//! nothing left to buy at any price — so a cheaper marginal term is a discount
//! on a product that does not exist.
//!
//! Run: `cargo run --release -p pixelflow-pipeline --example poly_precision`

use pixelflow_codegen::emit::compile;
use pixelflow_codegen::{JIT_VECTOR_BYTES, Point4, TileSlice};
use pixelflow_ir::passes::EXP2_POLY;
use pixelflow_ir::{Environment, ExprBuilder, ExprData, OpKind, Rooted, Term};
use pixelflow_pipeline::jit_bench::{BenchMode, BenchSession};
use pixelflow_pipeline::poly::{PolyForm, build, chebyshev_fit};

const LANES: usize = JIT_VECTOR_BYTES / 4;

/// Points the error is sampled at, spanning `[0, 1)`.
const SAMPLES: usize = 1024;

/// Highest degree fitted. Past this the monomial Vandermonde's conditioning
/// starts to matter, and the curve has been flat for six terms already.
const MAX_COEFFS: usize = 14;

const REPS: usize = 5;

/// `p(X · scale)` — `scale = 1/SAMPLES` sweeps `[0, 1)` for the error pass,
/// `1.0` for timing.
fn kernel(form: PolyForm, coeffs: &[f32], scale: f32) -> (Rooted<ExprData>, Environment) {
    let mut arena = ExprBuilder::new();
    let x = arena.push_var(0);
    let s = arena.push_const(scale);
    let arg = arena.push_binary(OpKind::Mul, x, s);
    let root = build(&mut arena, form, coeffs, arg);
    arena.finish(&[root])
}

/// Max `|JIT(x) − f(x)|` over the sampled range, using the JIT's own
/// arithmetic — FMA rounding included, which is the whole point: the floor
/// being looked for IS a rounding floor, so no scalar oracle can stand in.
fn error(form: PolyForm, coeffs: &[f32], f: impl Fn(f64) -> f64) -> f64 {
    let (expr, env) = kernel(form, coeffs, 1.0 / SAMPLES as f32);
    let code = compile(Term::new(expr.entry(), &env))
        .expect("compile")
        .code;
    let groups = SAMPLES / LANES;
    let mut out = vec![0.0f32; groups * LANES];
    let mut x0 = [0.0f32; LANES];
    for (i, lane) in x0.iter_mut().enumerate() {
        *lane = i as f32;
    }
    unsafe {
        code.call_collapse(
            core::ptr::null(),
            TileSlice::contiguous(out.as_mut_ptr(), groups, 1),
            Point4::new(x0, [0.0; LANES], [0.0; LANES], [0.0; LANES]),
        );
    }
    out.iter()
        .enumerate()
        .map(|(i, &got)| {
            let x = f64::from(i as f32 / SAMPLES as f32);
            (f64::from(got) - f(x)).abs()
        })
        .fold(0.0f64, f64::max)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    v[v.len() / 2]
}

fn scanline_ns(session: &mut BenchSession, form: PolyForm, coeffs: &[f32]) -> f64 {
    let (expr, env) = kernel(form, coeffs, 1.0);
    let term = Term::new(expr.entry(), &env);
    median(
        (0..REPS)
            .map(|_| {
                session
                    .benchmark_term(term, BenchMode::Scanline)
                    .expect("benchmark")
                    .ns
            })
            .collect(),
    )
}

fn main() {
    let f = |x: f64| x.exp2();
    let mut session = BenchSession::new();

    println!(
        "# accuracy per nanosecond — target exp2 on [0,1], LANES={LANES}, host fma={}",
        cfg!(target_feature = "fma"),
    );
    println!(
        "# err = max |JIT(x) − exp2(x)| in the JIT's own f32 arithmetic; \
         ns = per eval per lane, BenchMode::Scanline"
    );
    println!(
        "# passes::EXP2_POLY ships {} coefficients — the marker below shows where that sits",
        EXP2_POLY.len()
    );
    println!(
        "\n{:>3} {:>12} {:>12} {:>10} {:>10}",
        "n", "horner err", "estrin err", "H ns", "E ns"
    );

    let mut prev_err: Option<f64> = None;
    let mut knee: Option<usize> = None;
    for n in 2..=MAX_COEFFS {
        let coeffs = chebyshev_fit(f, n);
        let eh = error(PolyForm::Horner, &coeffs, f);
        let ee = error(PolyForm::Estrin, &coeffs, f);
        let hn = scanline_ns(&mut session, PolyForm::Horner, &coeffs);
        let en = scanline_ns(&mut session, PolyForm::Estrin, &coeffs);

        // The knee: the last degree that still bought a meaningful factor.
        // Past it the fit is no longer what limits accuracy — the format is.
        // 1.5× rather than 2× so a term that genuinely halves the error is not
        // called flat; the curve's own shape (30×, then 1.0×) means nothing
        // between those two thresholds actually occurs.
        const MEANINGFUL_GAIN: f64 = 1.5;
        let gain = prev_err.map(|p| p / eh);
        if knee.is_none() && gain.is_some_and(|g| g < MEANINGFUL_GAIN) {
            knee = Some(n - 1);
        }
        let mark = match (gain, n == EXP2_POLY.len()) {
            (_, true) => "← EXP2_POLY ships this degree".to_string(),
            (Some(g), _) if g < MEANINGFUL_GAIN => {
                format!("{g:.1}× — flat, f32 floor reached")
            }
            (Some(g), _) => format!("{g:.0}× better than n={}", n - 1),
            (None, _) => String::new(),
        };
        println!("{n:>3} {eh:>12.3e} {ee:>12.3e} {hn:>10.3} {en:>10.3}  {mark}");
        prev_err = Some(eh);
    }

    if let Some(k) = knee {
        let flat = chebyshev_fit(f, k);
        let long = chebyshev_fit(f, MAX_COEFFS);
        let cheap = scanline_ns(&mut session, PolyForm::Horner, &flat);
        let dear = scanline_ns(&mut session, PolyForm::Horner, &long);
        println!(
            "\n# knee at n={k}. Going to n={MAX_COEFFS} costs {:+.0}% time \
             ({cheap:.3} → {dear:.3} ns) and buys nothing.",
            100.0 * (dear / cheap - 1.0),
        );
        println!(
            "# So a schedule with a cheaper marginal term has nothing to spend \
             the discount on:\n# past the knee, accuracy is bounded by f32, not by degree."
        );
    }
}
