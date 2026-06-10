//! ASM inspection for circle SDF - Field arithmetic regression test
//!
//! Run: cargo show-asm -p pixelflow-graphics --example circle_asm circle_kernel --release

use pixelflow_core::{Field, ManifoldCompat, ManifoldExt, X, Y};
use std::hint::black_box;

/// Circle kernel: (x-cx)^2 + (y-cy)^2 < r^2 ? 1.0 : 0.0
/// This exercises Field mul, add, sub, and select
#[inline(never)]
#[must_use]
pub fn circle_kernel(x: Field, y: Field) -> Field {
    let cx = 128.0f32;
    let cy = 128.0f32;
    let r = 85.0f32;

    let dx = X - cx;
    let dy = Y - cy;
    let dist_sq = dx.clone() * dx + dy.clone() * dy;
    let inside = dist_sq.lt(r * r);
    inside
        .select(1.0f32, 0.0f32)
        .eval_raw(x, y, Field::from(0.0), Field::from(0.0))
}

fn main() {
    let x = Field::sequential(0.0);
    let y = Field::from(64.0);

    let result = circle_kernel(black_box(x), black_box(y));
    black_box(result);
}
