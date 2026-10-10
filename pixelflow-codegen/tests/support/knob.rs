// Which pipeline this process compiles with, for the tests that mean something
// under only one of them. Included, not compiled, like `sibling_rows.rs`.
//
// The knob is `PIXELFLOW_CODEGEN`, read once per process like the tier is.

/// Whether this process compiles with the selection pipeline.
fn selection() -> bool {
    std::env::var("PIXELFLOW_CODEGEN")
        .is_ok_and(|name| name.trim().eq_ignore_ascii_case("selection"))
}
