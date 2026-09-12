//! D1's **usefulness** half, against a mask built through the public
//! `Kernel` builder rather than by pushing arena nodes.
//!
//! `pixelflow-ir`'s own suite checks soundness by containment over the whole
//! extent, and does it on hand-built arenas — which pins the analysis but not
//! the *shape the builder actually emits*. If `Kernel::lt` or `Kernel::and`
//! lowered to something else tomorrow, that suite would stay green while
//! every production mask silently widened to the full extent: sound, and
//! worthless. This is the test that fails instead.
//!
//! The comparison target is the cell grid's `grid_range` — `[0, cols) ×
//! [0, rows)` — which is the one hand-written range the plan
//! ([one conditional, three lowerings](../../docs/plans/2026-09-08-one-conditional-three-lowerings.md) §8)
//! nominates. It is a comparison, not an oracle: the containment assertion
//! below is what decides correctness, and the equality is what decides
//! whether the derivation earns its keep.

use pixelflow_core::Kernel;
use pixelflow_ir::{LatticeShape, MaskSupport, mask_support};

/// The grid fence exactly as `pixelflow-core`'s cell grid writes it.
fn in_grid(width: f32, height: f32) -> Kernel {
    let k = Kernel::constant;
    let (x, y) = (Kernel::x(), Kernel::y());
    x.ge(&k(0.0))
        .and(&x.lt(&k(width)))
        .and(&y.ge(&k(0.0)))
        .and(&y.lt(&k(height)))
}

#[test]
fn the_grid_fence_derives_its_own_grid() {
    const FRAME: [u32; 2] = [128, 96];
    const GRID: [u32; 2] = [80, 48];

    let mask = in_grid(GRID[0] as f32, GRID[1] as f32);
    let (arena, root) = mask.parts();
    let shape = LatticeShape::new(FRAME);
    let got = mask_support(arena, root, shape);

    // Soundness first, and checked directly: every index the mask is nonzero
    // at must be inside. The mask is a conjunction of coordinate/literal
    // comparisons, so its truth at an integer index is exactly this.
    for y in 0..FRAME[1] {
        for x in 0..FRAME[0] {
            let nonzero = x < GRID[0] && y < GRID[1];
            assert!(
                !nonzero || got.contains([x, y]),
                "derived support excludes ({x}, {y}), where the fence is nonzero"
            );
        }
    }

    // Usefulness: it must be the grid, not the frame. `grid_range` for this
    // metric is `IndexRange::new(0, 0, 80, 48)`.
    assert_eq!(got.start(), [0, 0]);
    assert_eq!(
        got.end(),
        GRID,
        "the fence must derive the grid; deriving the frame would be sound and worthless"
    );
    assert_ne!(
        got,
        MaskSupport::everywhere(shape),
        "a derivation that widens to the whole lattice has bought nothing"
    );
}

/// A grid that fills its frame derives the frame, and that is not a failure —
/// it is the same rectangle by arithmetic rather than by giving up. Pinned so
/// the `assert_ne!` above is never read as "narrower is always better".
#[test]
fn a_full_frame_grid_derives_the_frame() {
    const FRAME: [u32; 2] = [32, 32];
    let mask = in_grid(FRAME[0] as f32, FRAME[1] as f32);
    let (arena, root) = mask.parts();
    let shape = LatticeShape::new(FRAME);
    let got = mask_support(arena, root, shape);

    assert_eq!(got.start(), [0, 0]);
    assert_eq!(got.end(), FRAME);
}
