//! What the selection pipeline emits grows with the kernel, not faster.
//!
//! Its register scan is linear in the instructions it is given times the size
//! of a register file, so its time is bounded by how many instructions there
//! are. These bound that number through what a compile reports: the
//! instructions a kernel schedules, the loads, stores and pool reads the
//! allocator adds to them, and the bytes of code they come to.
//!
//! The count of selected machine instructions is not reported, only the
//! scheduled operations they were selected from (`ScopeTraffic::instructions`,
//! which counts a constant not at all: it is a pool read, where it is read), so
//! the code's bytes stand in for it. They run on the selection pipeline, a tier's
//! default or `PIXELFLOW_CODEGEN=selection`; under the legacy pipeline they do
//! nothing.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::emit::{CompileResult, ScopeTraffic, compile};
use pixelflow_codegen::jit_vector_bytes;
use pixelflow_ir::fold::{Binder, Fold, Monoid};
use pixelflow_ir::{ExprArena, ExprId, LatticeShape, OpKind};

mod rows {
    include!("support/sibling_rows.rs");
}
include!("support/knob.rs");

const BYTES_PER_LANE: usize = 4;

/// A kernel family: the kernel at a term count.
type Family = fn(usize) -> (ExprArena, ExprId);

/// What a compile adds to the operations it was given, and what it came to.
#[derive(Clone, Copy, Debug)]
struct Size {
    /// Scheduled operations selected, constants excluded.
    scheduled: u64,
    /// Loads, stores and pool reads the allocator placed.
    inserted: u64,
    bytes: u64,
}

impl Size {
    fn of(result: &CompileResult) -> Self {
        let sum = |count: fn(&ScopeTraffic) -> u64| -> u64 {
            result.traffic.scopes.iter().map(count).sum()
        };
        Self {
            scheduled: sum(|s| s.instructions),
            inserted: sum(|s| s.loads + s.stores + s.remats),
            bytes: result.code.as_bytes().len() as u64,
        }
    }
}

fn compiled(arena: &ExprArena, root: ExprId, columns: u32) -> CompileResult {
    compile(arena, root, LatticeShape::new([columns, rows::ROWS])).expect("the kernel compiles")
}

/// The allocator adds at most two instructions for each the driver selected.
/// The driver selects at least one for each scheduled operation, so this holds
/// the allocator to twice the scheduled operations, a bound that fails earlier.
#[test]
fn the_allocator_inserts_at_most_twice_what_was_scheduled() {
    if !selection() {
        return;
    }
    const PER_OPERATION: u64 = 2;
    let lanes = (jit_vector_bytes() / BYTES_PER_LANE) as u32;
    for row in &rows::TABLE {
        let (arena, root) = (row.build)();
        let size = Size::of(&compiled(&arena, root, row.width.columns(lanes)));
        assert!(
            size.inserted <= PER_OPERATION * size.scheduled,
            "{}: {} loads, stores and pool reads were inserted among {} scheduled operations",
            row.name,
            size.inserted,
            size.scheduled
        );
    }
}

/// Twice the terms is at most a little over twice the code and twice the
/// insertions. This bounds what is emitted, not the time spent emitting it: a
/// scan that stored every value to every slot would show as four times, and
/// one that rescanned the kernel for each instruction and emitted the same
/// code would not show at all.
#[test]
fn doubling_a_kernel_at_most_doubles_what_is_emitted() {
    if !selection() {
        return;
    }
    /// Past linear by the rounding a fixed prologue and a square-root number of
    /// shared products add.
    const GROWTH_PERCENT: u64 = 220;
    const TERMS: usize = 256;
    let families: [(&str, Family); 2] = [
        ("parked_roots", |n| rows::parked_roots(n as u64)),
        ("deep_frame", rows::deep_frame),
    ];
    for (name, family) in families {
        let at = |terms| {
            let (arena, root) = family(terms);
            Size::of(&compiled(&arena, root, rows::REMAINDER_WIDTH))
        };
        let (small, large) = (at(TERMS), at(2 * TERMS));
        for (what, small, large) in [
            ("scheduled operations", small.scheduled, large.scheduled),
            ("inserted instructions", small.inserted, large.inserted),
            ("bytes of code", small.bytes, large.bytes),
        ] {
            assert!(
                large * 100 <= small * GROWTH_PERCENT,
                "{name}: {TERMS} terms have {small} {what}, and twice the terms {large}"
            );
        }
    }
}

/// Selection picks at most four instructions for each scheduled operation and
/// an instruction is at most ten bytes, so a row's code is bounded by its
/// scheduled operations: the per-row stand-in for selected instructions at
/// most four times the scheduled ones. The rows come to 11 to 27.
#[test]
fn a_row_emits_bytes_in_proportion_to_its_scheduled_operations() {
    if !selection() {
        return;
    }
    const BYTES_PER_OPERATION: u64 = 4 * 10;
    let lanes = (jit_vector_bytes() / BYTES_PER_LANE) as u32;
    for row in &rows::TABLE {
        let (arena, root) = (row.build)();
        let size = Size::of(&compiled(&arena, root, row.width.columns(lanes)));
        assert!(
            size.bytes <= BYTES_PER_OPERATION * size.scheduled,
            "{}: {} bytes of code for {} scheduled operations",
            row.name,
            size.bytes,
            size.scheduled
        );
    }
}

/// A chain of `n` arithmetic ops over the two coordinates, each reading the
/// one before it: `x + y - y - y - ...`.
fn chain(n: usize) -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let mut acc = a.push_binary(OpKind::Add, x, y);
    for _ in 1..n {
        acc = a.push_binary(OpKind::Sub, acc, y);
    }
    (a, acc)
}

/// One more op scheduled is one more counted, whatever it schedules
/// alongside: the lattice's own folds wrap every root alike, so a chain's
/// scheduled count over its total is the lattice's fixed overhead plus one
/// per link, and the overhead cancels in the difference. Catches a selector
/// that counts a scheduled operation by the wrong rule (one that counts only
/// the ops it should skip, or an accumulator a multiply can never move off
/// zero): either leaves this difference at 0, not 1.
#[test]
fn one_more_scheduled_operation_moves_the_total_scheduled_count_by_one() {
    if !selection() {
        return;
    }
    let lanes = (jit_vector_bytes() / BYTES_PER_LANE) as u32;
    let scheduled_at = |n| {
        let (arena, root) = chain(n);
        Size::of(&compiled(&arena, root, lanes)).scheduled
    };
    let (three, four) = (scheduled_at(3), scheduled_at(4));
    assert_eq!(
        four,
        three + 1,
        "a chain of 3 ops schedules {three}; one more op should schedule exactly one more, \
         not {four}"
    );
}

/// Two sibling folds schedule two more operations than one does: the second
/// `Reduce` opening its own loop, and the `Add` that combines the two
/// results — both counted the same way every other scheduled operation is.
/// Catches a selector that fails to count a fold when it opens one, which a
/// bound on `inserted`/`bytes` cannot see (a fold's own body is unaffected,
/// so nothing downstream grows or shrinks).
#[test]
fn a_second_sibling_fold_moves_the_total_scheduled_count_by_two() {
    if !selection() {
        return;
    }
    let lanes = (jit_vector_bytes() / BYTES_PER_LANE) as u32;
    let one = {
        let mut a = ExprArena::new();
        let binder = Binder::from_slot(0).expect("slot 0");
        let k = a.push_var(binder.var());
        let root = a.push_reduce(Fold::new(Monoid::SUM, binder, 0..3), k);
        Size::of(&compiled(&a, root, lanes)).scheduled
    };
    let two = {
        let mut a = ExprArena::new();
        let b0 = Binder::from_slot(0).expect("slot 0");
        let b1 = Binder::from_slot(1).expect("slot 1");
        let k0 = a.push_var(b0.var());
        let fold0 = a.push_reduce(Fold::new(Monoid::SUM, b0, 0..3), k0);
        let k1 = a.push_var(b1.var());
        let fold1 = a.push_reduce(Fold::new(Monoid::SUM, b1, 0..4), k1);
        let root = a.push_binary(OpKind::Add, fold0, fold1);
        Size::of(&compiled(&a, root, lanes)).scheduled
    };
    assert_eq!(
        two,
        one + 2,
        "one fold schedules {one}; a second sibling fold plus the Add joining them should \
         schedule exactly two more, not {two}"
    );
}
