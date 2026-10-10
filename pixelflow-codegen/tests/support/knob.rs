// Which pipeline this process compiles with, for the tests that mean something
// under only one of them. Included, not compiled, like `sibling_rows.rs`.
//
// The knob is `PIXELFLOW_CODEGEN`, read once per process like the tier is. When
// it is unset a tier compiles with its own pipeline: selection on AVX2, the
// legacy emitters on the tiers that have not switched.

/// Whether this process compiles with the selection pipeline.
fn selection() -> bool {
    match std::env::var("PIXELFLOW_CODEGEN") {
        Ok(name) => name.trim().eq_ignore_ascii_case("selection"),
        Err(_) => pixelflow_codegen::isa::detect() == pixelflow_codegen::isa::Isa::Avx2,
    }
}
