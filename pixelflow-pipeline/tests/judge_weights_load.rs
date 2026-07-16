//! Verifies the Judge extraction-head weights round-trip through the
//! production serializer: `ExprNnue::save` (which writes the current "TRID"
//! format) followed by `ExprNnue::from_bytes` (which only accepts TRID).
//!
//! Phase 2 of docs/plans/2026-07-07-guided-saturation-redesign.md: the
//! shipped state before the retrain was NO weights (old TRIC-format file
//! deleted; the compiler falls back to an explicit zero model). This test is
//! the falsifiable check that a freshly-produced weights file is not dead on
//! arrival — the magic matches AND the loaded model carries real signal.
//!
//! It is deliberately self-contained: it produces its own fixture via
//! `new_with_latency_prior` rather than reading a checked-in/gitignored
//! `.bin`, so it passes on a fresh checkout and in CI. Whether an *actually
//! retrained* model beats the latency prior is the job of the
//! `bench_extraction_3way` benchmark, not a unit test.

use pixelflow_search::nnue::factored::ExprNnue;

#[test]
fn judge_weights_round_trip_via_trid() {
    // A latency-prior-initialized model is well-formed and non-zero — the
    // same code path `bootstrap_extraction_head` starts from before training.
    let model = ExprNnue::new_with_latency_prior(0xF00D_2026);

    // Save through the production path to a unique temp file.
    let mut path = std::env::temp_dir();
    path.push(format!("pf_judge_trid_{}.bin", std::process::id()));
    model
        .save(&path)
        .unwrap_or_else(|e| panic!("ExprNnue::save failed for {}: {e}", path.display()));

    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let _ = std::fs::remove_file(&path);

    // The magic must be TRID (not a stale TRIC/TRIB), and the loader must accept it.
    assert_eq!(&bytes[0..4], b"TRID", "saved file is not TRID-format");
    let loaded = ExprNnue::from_bytes(&bytes)
        .unwrap_or_else(|e| panic!("ExprNnue::from_bytes rejected a freshly-saved TRID file: {e}"));

    // Beyond "magic matched": a model whose embeddings are all zero/non-finite
    // would still parse as valid TRID but carry no signal. Guard against a
    // silently-dead file.
    let has_signal = loaded
        .embeddings
        .e
        .iter()
        .flatten()
        .any(|&v| v.is_finite() && v != 0.0);
    assert!(
        has_signal,
        "round-tripped model has an all-zero/non-finite embedding table"
    );
}
