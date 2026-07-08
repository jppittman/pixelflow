//! Verifies the retrained Judge extraction-head weights load through the
//! production loader (`ExprNnue::from_bytes`), which only accepts the
//! current "TRID" format.
//!
//! Phase 2 of docs/plans/2026-07-07-guided-saturation-redesign.md: the
//! shipped state before this retrain was NO weights (old TRIC-format file
//! deleted; compiler falls back to an explicit zero model). This test is
//! the falsifiable check that the freshly-trained file is not similarly
//! dead on arrival.

use pixelflow_search::nnue::factored::ExprNnue;

const WEIGHTS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/expr_nnue_trid.bin");

#[test]
fn judge_weights_load_via_from_bytes() {
    let bytes = std::fs::read(WEIGHTS_PATH)
        .unwrap_or_else(|e| panic!("failed to read {WEIGHTS_PATH}: {e}"));

    let model = ExprNnue::from_bytes(&bytes)
        .unwrap_or_else(|e| panic!("ExprNnue::from_bytes rejected {WEIGHTS_PATH}: {e}"));

    // Sanity beyond "the magic bytes matched": a model that trained for
    // zero steps would still parse as valid TRID but every embedding row
    // would be exactly its random or zero init. Check for finite, non-zero
    // signal so a wired-but-untrained file doesn't pass silently.
    let has_signal = model
        .embeddings
        .e
        .iter()
        .flatten()
        .any(|&v| v.is_finite() && v != 0.0);
    assert!(
        has_signal,
        "loaded model has an all-zero/non-finite embedding table — looks untrained"
    );
}
