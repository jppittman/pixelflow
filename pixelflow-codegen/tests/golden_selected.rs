//! **The selection pipeline's code, byte for byte.**
//!
//! `GOLDEN_SELECTED` is `emit::tests::sibling_folds`'s `GOLDEN` for the new
//! pipeline: the length and FNV-1a 64 digest of what each of `rows::TABLE`'s
//! kernels compiles to. It is born with AVX2, the one tier the pipeline has a
//! backend for (and compiles with), and gains AVX-512 and NEON as they get
//! theirs. It obeys the same rule from birth:
//!
//! **A refactor does not edit this table; an intentional byte change does, in
//! a commit of its own that says why.** A commit that edits it beside other
//! work cannot be told apart from one that moved bytes by accident, which is
//! the thing it exists to catch. When it fails, the failure prints the whole
//! recomputed table.
//!
//! Unlike `GOLDEN`, which emits all three targets from any host through the
//! crate's own entry points, this reaches the pipeline through `compile`, so it
//! is the host's tier and the process's knob: it does nothing unless run on the
//! selection pipeline at `PIXELFLOW_ISA=avx2`, where it is the default. The
//! `isa-matrix` job does. `GOLDEN`'s AVX2 column pins the same code.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::emit::compile;
use pixelflow_codegen::{fnv1a64, jit_vector_bytes};
use pixelflow_ir::LatticeShape;

mod rows {
    include!("support/sibling_rows.rs");
}
include!("support/knob.rs");

const BYTES_PER_LANE: usize = 4;

/// A row's emitted code: its length in bytes and the FNV-1a 64 digest of those
/// bytes.
type Bytes = (usize, u64);

/// `rows::TABLE`'s bytes through the selection pipeline, on AVX2. A row's
/// provenance is `git log -L` on it.
const GOLDEN_SELECTED_AVX2: [Bytes; 11] = [
    (580, 0xcd22a56b8b260b1a),
    (568, 0x86d60c1545a26d56),
    (880, 0x3ebb4798982a8c6d),
    (896, 0x609dcba2dd47b291),
    (204768, 0x3bee772bb0d20234),
    (2292, 0x6f731e42104d8439),
    (992, 0x99dfdad6d446b32d),
    (852, 0x2c228ca476b28b53),
    (556, 0x8403cb1c517f09a5),
    (452, 0x737225ee0861c2c9),
    (356060, 0x3046d9022ee419bb),
];

#[test]
fn the_sibling_fold_rows_emit_the_recorded_bytes_through_selection() {
    if !selection() {
        return;
    }
    let lanes = (jit_vector_bytes() / BYTES_PER_LANE) as u32;
    let mut recomputed = Vec::new();
    let mut moved = Vec::new();
    for (row, pin) in rows::TABLE.iter().zip(GOLDEN_SELECTED_AVX2) {
        let (arena, root) = (row.build)();
        let shape = LatticeShape::new([row.width.columns(lanes), rows::ROWS]);
        let result = compile(&arena, root, shape).expect("a sibling-fold row compiles");
        let code = result.code.as_bytes();
        let actual = (code.len(), fnv1a64(code));
        if actual != pin {
            moved.push(format!(
                "{}: pinned {pin:x?}, emitted {actual:x?}",
                row.name
            ));
        }
        recomputed.push(format!("    ({}, {:#018x}),", actual.0, actual.1));
    }
    assert!(
        moved.is_empty(),
        "emitted bytes moved from GOLDEN_SELECTED_AVX2:\n{}\n\n\
         if the change is intentional, re-baseline it in its own commit with:\n{}",
        moved.join("\n"),
        recomputed.join("\n")
    );
}
