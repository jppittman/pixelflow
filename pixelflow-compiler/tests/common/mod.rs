//! What several integration tests read the same way.
//!
//! Each test binary uses a subset, so an unused helper is not a defect here.
#![allow(dead_code)]

use pixelflow_core::{Kernel, Lattice};
use pixelflow_ir::key::canonical;

/// The sample the values are read at: `X = 3`, `Y = 5`, so every binding in a
/// body has a distinct value.
pub const AT: (f32, f32) = (3.0, 5.0);

/// A kernel's value at [`AT`].
pub fn bake(k: &Kernel) -> f32 {
    Lattice::eval_at(k, AT.0, AT.1)
}

pub fn x() -> Kernel {
    Kernel::x()
}

pub fn y() -> Kernel {
    Kernel::y()
}

/// Whether two kernels are one program: the same canonical key, which is what
/// the JIT cache keys on. The tables beside it say which argument each slot
/// binds, and two instances of one argument are two identities.
pub fn assert_same_program(written: &Kernel, built: &Kernel) {
    let (written_arena, written_root) = written.parts();
    let (built_arena, built_root) = built.parts();
    assert_eq!(
        canonical(written_arena, written_root).key,
        canonical(built_arena, built_root).key,
        "written: {}\nbuilt:   {}",
        written_arena.display(written_root),
        built_arena.display(built_root),
    );
}
