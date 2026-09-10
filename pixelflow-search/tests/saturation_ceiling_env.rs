//! `PIXELFLOW_SATURATION_CEILING_MS` — the diagnostic override on
//! `Optimizer::production()`'s wall-clock safety ceiling
//! (`docs/plans/2026-09-01-production-budget-determinism.md`).
//!
//! The env var is process-global, so every test here holds [`ENV_LOCK`] for
//! its whole body — the default parallel test runner would otherwise let
//! two of these tests interleave and each observe the other's value.

use std::sync::Mutex;

use pixelflow_ir::OpKind;
use pixelflow_ir::Rooted;
use pixelflow_ir::expr::{Environment, ExprBuilder, ExprData, Term};
use pixelflow_search::egraph::Optimizer;

static ENV_LOCK: Mutex<()> = Mutex::new(());

const VAR: &str = "PIXELFLOW_SATURATION_CEILING_MS";

/// A classical-tier expression (200+ nodes) substantial enough that
/// saturating it measurably exceeds 1ms even release-optimized, and by a
/// wide margin in the unoptimized `dev` profile `cargo test` runs under —
/// the same opt-level-0 regime the calibration doc's whole argument is
/// about.
fn fixture() -> (Rooted<ExprData>, Environment) {
    let mut a = ExprBuilder::new();
    let mut cur = a.push_var(0);
    for i in 0..40u32 {
        let v = a.push_var((i % 4) as u8);
        let c = a.push_const(1.0 + i as f32 * 0.01);
        let sum = a.push_binary(OpKind::Add, cur, v);
        let prod = a.push_binary(OpKind::Mul, sum, c);
        cur = a.push_binary(OpKind::Sub, prod, v);
    }
    a.finish(&[cur])
}

/// A structural rendering of the extracted DAG. `expr::encode` is already the
/// canonical serialization, so this is its bytes rather than a second answer
/// to "same term?".
fn arena_shape(g: &(Rooted<ExprData>, Environment)) -> String {
    format!("{:?}", pixelflow_ir::encode(g.0.entry()))
}

fn optimize(input: &(Rooted<ExprData>, Environment)) -> (Rooted<ExprData>, Environment) {
    let mut optimizer = Optimizer::production();
    let mut eg = optimizer.egraph();
    let root_class = pixelflow_search::egraph::insert_term(
        Term::new(input.0.entry(), &input.1),
        &mut eg,
        pixelflow_search::egraph::Vocabulary::Templates,
    )
    .expect("insert into e-graph");
    let optimized = optimizer.run(&mut eg, root_class, input.0.len());
    optimized.to_rooted(&eg, root_class)
}

/// The invariant the calibration doc states outright: the override can only
/// change *whether `Optimizer::run` panics*, never what it computes. A tiny
/// override must panic; a generous one, `off`, and unset must all succeed
/// and must all extract byte-identical terms.
#[test]
fn the_override_changes_only_whether_run_panics_never_what_it_computes() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let input = fixture();

    // SAFETY: `_guard` holds this test's exclusive claim on the env var for
    // the whole function body — no other test in this binary touches it
    // without first taking the same lock.
    unsafe { std::env::remove_var(VAR) };
    let baseline = optimize(&input);

    unsafe { std::env::set_var(VAR, "1") };
    let panicked =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| optimize(&input))).is_err();
    assert!(
        panicked,
        "a 1ms PIXELFLOW_SATURATION_CEILING_MS must panic on a classical-tier run"
    );

    unsafe { std::env::set_var(VAR, "600000") };
    let generous = optimize(&input);

    unsafe { std::env::set_var(VAR, "off") };
    let disabled = optimize(&input);

    unsafe { std::env::remove_var(VAR) };

    assert_eq!(
        arena_shape(&baseline),
        arena_shape(&generous),
        "a generous ceiling override must not change the extracted term"
    );
    assert_eq!(
        arena_shape(&baseline),
        arena_shape(&disabled),
        "disabling the ceiling must not change the extracted term"
    );
}

/// An unparsable value fails loudly rather than silently picking a default —
/// the no-silent-failures half of the override's contract.
#[test]
fn an_unparsable_override_panics_rather_than_picking_a_default() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let input = fixture();

    unsafe { std::env::set_var(VAR, "banana") };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| optimize(&input)));
    unsafe { std::env::remove_var(VAR) };

    assert!(
        result.is_err(),
        "an unparsable PIXELFLOW_SATURATION_CEILING_MS must panic, not silently pick a default"
    );
}
