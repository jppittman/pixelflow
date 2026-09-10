//! S0 of `docs/plans/2026-09-09-glyph-as-a-fold-execution.md` — U1/U2.
//!
//! The glyph-as-a-fold rewrite wants a `Kernel` that reads a host-built table
//! at exact integer indices where the ROW index is a `Kernel::over` reduce
//! binder (`table.at(&constant(col), &binder)`), so a glyph's per-piece
//! coefficients can live in one bound buffer instead of `extent`-many
//! separately-compiled constants. That combination — a buffer gather whose
//! index is a reduce binder, carried through `.at()`, unrolled by
//! `legalize`, and executed by the JIT — exists nowhere else in the
//! workspace, so this file proves it before anything is built on it.
//!
//! This lives in `pixelflow-core` and not `pixelflow-codegen` or
//! `pixelflow-ir`: the test needs both `Kernel::over` (defined in
//! `pixelflow-ir`) and the compiled `Manifold`/`Lattice::collapse` pipeline
//! that turns a bound buffer into numbers (defined in `pixelflow-core`, on
//! top of `pixelflow-codegen`'s JIT). `pixelflow-core` is the only crate
//! that depends on both without a cycle: `pixelflow-codegen` depends on
//! `pixelflow-ir` but knows nothing of `Manifold`/`DiscreteManifold`, and
//! `pixelflow-ir` depends on neither. `pixelflow-compiler` also sees both
//! (through a dev-dependency on `pixelflow-core`), but its purpose is the
//! `kernel!` macro front end, not the runtime composition surface this test
//! exercises — `pixelflow-core` is where a bound-buffer-plus-binder read
//! belongs as a fact about the language, not about macro expansion.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use std::sync::Arc;

use pixelflow_core::{DiscreteManifold, Kernel, Lattice, Manifold};
use pixelflow_ir::arena::BufferIdentity;

/// Columns in the host table — distinct per column so a transposed
/// (row, col) swap cannot pass by accident.
const TABLE_COLS: usize = 4;
/// Rows in the host table. Deliberately unequal to `TABLE_COLS`: a
/// transposition bug would then hit a different declared extent on the
/// wrong axis and clamp to a wrong-but-plausible index instead of silently
/// reading the right memory in the wrong order.
const TABLE_ROWS: usize = 3;

/// The host table, row-major as `DiscreteManifold` addresses it
/// (`buffer[row * width + col]`). Every value is distinct and, within a
/// column, in no particular order across rows — a `min_over` that is exact
/// only for the *first* row of a column, or a `sum_over` that silently
/// drops or repeats one, would still pass a monotone table.
const TABLE: [[f32; TABLE_COLS]; TABLE_ROWS] = [
    [3.0, 40.0, 500.0, 6000.0],
    [70.0, 8.0, 90.0, 100.0],
    [2.0, 300.0, 4.0, 50.0],
];

/// Flatten [`TABLE`] row-major into the `Vec<f32>` a buffer binds.
fn flatten_table() -> Vec<f32> {
    TABLE.iter().flatten().copied().collect()
}

/// Mint a fresh buffer identity, bind [`TABLE`] to it, and return both the
/// identity/data pair ([`Manifold::bind`]'s shape) and the single-tap gather
/// [`Kernel`] that reads it ([`DiscreteManifold::kernel_for`] — nearest-
/// neighbour, exact at in-range integer indices per the crate's own
/// `floor`-then-`clamp` addressing).
fn bind_table() -> ((BufferIdentity, Arc<Vec<f32>>), Kernel) {
    let id = BufferIdentity::mint();
    let data = Arc::new(flatten_table());
    let table = DiscreteManifold::kernel_for(id, TABLE_COLS as u32, TABLE_ROWS as u32);
    ((id, data), table)
}

/// Compile `kernel` (which must read nothing but the buffer in `binding`),
/// bind it, and collapse it to the single resulting scalar. The kernel is
/// independent of X/Y here — the reduce binder is the only varying index —
/// so a 1x1 lattice is enough to pull the one value out.
fn collapse_scalar(kernel: &Kernel, binding: (BufferIdentity, Arc<Vec<f32>>)) -> f32 {
    let program = Manifold::compile(kernel, [1, 1]);
    let bound = program.bind(&[binding]);
    let out = Lattice::frame(1, 1).collapse(&bound);
    out.buffer()[0]
}

/// `Σ_{row} table[row][col]`, computed by plain host iteration — independent
/// of the kernel path, so a match is evidence the two agree rather than a
/// tautology against a hand-copied constant.
fn host_column_sum(col: usize) -> f32 {
    TABLE.iter().map(|row| row[col]).sum()
}

/// `min_{row} table[row][col]`, computed by plain host iteration.
fn host_column_min(col: usize) -> f32 {
    TABLE
        .iter()
        .map(|row| row[col])
        .fold(f32::INFINITY, f32::min)
}

/// `Σ_{row} table[row][col] * row`, computed by plain host iteration. The
/// binder-dependent weight catches an off-by-one or index-drift bug that an
/// unweighted sum cannot: dropping row 0 or duplicating row `N-1` still sums
/// to a plausible-looking total in the unweighted case, but shifts this
/// value by a term that depends on *which* row was mishandled.
fn host_column_weighted_sum(col: usize) -> f32 {
    TABLE
        .iter()
        .enumerate()
        .map(|(row, cols)| cols[col] * row as f32)
        .sum()
}

/// `Kernel::sum_over` a bound buffer's column, at every column, is exact.
///
/// `table.at(&constant(col), &i)` — the row index comes from the
/// `Kernel::over` binder, the column from a compile-time constant. Every
/// column is checked, not just one, so a bug that reads (row, col) as
/// (col, row) fails here rather than accidentally cancelling out.
#[test]
fn sum_over_reads_every_column_exactly() {
    for col in 0..TABLE_COLS {
        let (binding, table) = bind_table();
        let kernel = Kernel::sum_over(TABLE_ROWS as u32, |i| {
            table.at(&Kernel::constant(col as f32), i)
        });
        let got = collapse_scalar(&kernel, binding);
        let want = host_column_sum(col);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "sum_over column {col}: got {got}, want {want}"
        );
    }
}

/// `Kernel::min_over` a bound buffer's column, at every column, is exact.
/// [`TABLE`]'s per-column minimum sits at a different row for each column
/// (row 2 for column 0, row 1 for column 1, row 2 for column 2, row 2 for
/// column 3 — deliberately not always row 0), so this cannot pass by only
/// ever reading the first row correctly.
#[test]
fn min_over_reads_every_column_exactly() {
    for col in 0..TABLE_COLS {
        let (binding, table) = bind_table();
        let kernel = Kernel::min_over(TABLE_ROWS as u32, |i| {
            table.at(&Kernel::constant(col as f32), i)
        });
        let got = collapse_scalar(&kernel, binding);
        let want = host_column_min(col);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "min_over column {col}: got {got}, want {want}"
        );
    }
}

/// A `sum_over` whose body multiplies the gathered value by the binder
/// itself (`table.at(&c, i).mul(i)`) is exact, at every column. This is the
/// off-by-one/float-index-drift check: if the binder reaching the gather's
/// row index were shifted, truncated, or read as a different slot than the
/// one multiplying the value, the weighted sum would diverge from the
/// unweighted one in a way a plain `sum_over` cannot expose.
#[test]
fn binder_weighted_sum_over_reads_every_column_exactly() {
    for col in 0..TABLE_COLS {
        let (binding, table) = bind_table();
        let kernel = Kernel::sum_over(TABLE_ROWS as u32, |i| {
            table.at(&Kernel::constant(col as f32), i).mul(i)
        });
        let got = collapse_scalar(&kernel, binding);
        let want = host_column_weighted_sum(col);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "weighted sum_over column {col}: got {got}, want {want}"
        );
    }
}

/// Reports the arena's node count before and after `legalize`, at two
/// extents, so the unroll factor `legalize` performs on a
/// buffer-under-binder reduce is visible rather than assumed.
///
/// `legalize` runs `expand_reduce` before `expand_gather` (established fact,
/// confirmed by reading `pixelflow_ir::passes`): the `Reduce` node unrolls
/// into `extent` inlined copies of its body *first*, each with the binder
/// substituted as a distinct `Const`, and only then does each copy's now-
/// constant-indexed `Gather` lower to index arithmetic. So node count should
/// scale with `extent`, not stay flat — that scaling, not just "it doesn't
/// panic", is what this test checks and reports.
#[test]
fn legalize_unrolls_the_binder_and_the_gather_together() {
    /// A second extent, larger than [`TABLE_ROWS`], purely to see the unroll
    /// factor scale — this kernel is never collapsed, so it does not matter
    /// that most of its binder range falls outside the table's declared
    /// height (the addressing clamp handles that at runtime; `legalize` is a
    /// structural rewrite and never executes the gather).
    const WIDE_EXTENT: u32 = 16;

    let node_counts = |extent: u32| {
        let (_binding, table) = bind_table();
        let kernel = Kernel::sum_over(extent, |i| table.at(&Kernel::constant(0.0), i));
        let (arena, root) = kernel.parts();
        let before = arena.nodes_raw().len();
        let (legalized, _root) = pixelflow_ir::passes::legalize(arena, root).expect("legalize");
        let after = legalized.nodes_raw().len();
        (before, after)
    };

    let (before_3, after_3) = node_counts(TABLE_ROWS as u32);
    let (before_16, after_16) = node_counts(WIDE_EXTENT);

    eprintln!(
        "extent {}: {before_3} nodes before legalize, {after_3} after \
         (x{:.2})",
        TABLE_ROWS,
        after_3 as f64 / before_3 as f64
    );
    eprintln!(
        "extent {WIDE_EXTENT}: {before_16} nodes before legalize, {after_16} \
         after (x{:.2})",
        after_16 as f64 / before_16 as f64
    );

    // The unrolled reduce plus its now-constant-indexed gathers must contain
    // strictly more nodes than the pre-legalize arena (one `Reduce` node and
    // one `Gather` node, versus `extent` inlined, index-lowered copies of
    // each).
    assert!(
        after_3 > before_3,
        "extent {TABLE_ROWS}: legalize did not grow the arena \
         ({before_3} -> {after_3})"
    );
    assert!(
        after_16 > before_16,
        "extent {WIDE_EXTENT}: legalize did not grow the arena \
         ({before_16} -> {after_16})"
    );
    // A wider extent must unroll to more nodes than a narrower one, over the
    // same body — the concrete evidence that the unroll factor tracks
    // `extent` rather than being some fixed overhead.
    assert!(
        after_16 > after_3,
        "wider extent did not unroll to more nodes: extent {} -> {after_3}, \
         extent {WIDE_EXTENT} -> {after_16}",
        TABLE_ROWS
    );
}
