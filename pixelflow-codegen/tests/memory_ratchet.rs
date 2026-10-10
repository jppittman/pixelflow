//! The selection pipeline moves no more memory than the legacy one did.
//!
//! The quantity is what a call executes: each scope's stack loads, stack stores
//! and constant-pool reads, weighted by how many times the call runs the scope
//! (`EmitTraffic::trips`). A pool read is a memory read wherever it is spelled
//! (a broadcast the allocator placed, a constant a fold's trip test loads), and
//! the two pipelines count it the same way, so the sums are of like things.
//!
//! The legacy pipeline is deleted when the selection pipeline becomes the only
//! one, and the knob is read once per process, so its numbers are pinned here
//! and checked against the pipeline that made them whenever that runs. A change
//! that moves them is a change to the metric (or to the legacy pipeline), and
//! fails `legacy_traffic_is_the_recorded_table` with the table to paste.
//!
//! Run under `PIXELFLOW_CODEGEN=selection PIXELFLOW_ISA=avx2`, the ratchet holds
//! the new pipeline to that table; the legacy test runs under
//! `PIXELFLOW_ISA=avx2` alone. Each does nothing under the other.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::emit::{CompileResult, EmitTraffic, compile};
use pixelflow_codegen::jit_vector_bytes;
use pixelflow_ir::LatticeShape;

mod rows {
    include!("support/sibling_rows.rs");
}
include!("support/knob.rs");

/// The vector width of the AVX2 tier, the one the tables are for.
const AVX2_VECTOR_BYTES: usize = 32;
const BYTES_PER_LANE: usize = 4;

/// What one row costs: its code, and what a call executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cost {
    bytes: u64,
    loads: u64,
    stores: u64,
    pool_reads: u64,
}

impl Cost {
    /// The quantity the ratchet bounds: every memory read and write a call
    /// executes on the kernel's behalf.
    fn memory(self) -> u64 {
        self.loads + self.stores + self.pool_reads
    }

    fn of(result: &CompileResult) -> Self {
        let EmitTraffic { scopes, trips, .. } = &result.traffic;
        let weighted = |count: fn(&pixelflow_codegen::emit::ScopeTraffic) -> u64| -> u64 {
            scopes.iter().zip(trips).map(|(s, t)| count(s) * t).sum()
        };
        Self {
            bytes: result.code.as_bytes().len() as u64,
            loads: weighted(|s| s.loads),
            stores: weighted(|s| s.stores),
            pool_reads: weighted(|s| s.remats),
        }
    }
}

/// `row` compiled over the host tier's lattice.
fn compiled(row: &rows::Row) -> CompileResult {
    let lanes = (jit_vector_bytes() / BYTES_PER_LANE) as u32;
    let (arena, root) = (row.build)();
    let shape = LatticeShape::new([row.width.columns(lanes), rows::ROWS]);
    compile(&arena, root, shape).expect("a sibling-fold row compiles")
}

fn costs() -> Vec<Cost> {
    rows::TABLE
        .iter()
        .map(|row| Cost::of(&compiled(row)))
        .collect()
}

/// The legacy pipeline's cost for each of `rows::TABLE`, on AVX2.
///
/// To regenerate, run `legacy_traffic_is_the_recorded_table` under
/// `PIXELFLOW_ISA=avx2` with the default pipeline: its failure prints this
/// table. A row changes only in a commit that says why the legacy pipeline,
/// or what is counted, changed.
///
/// The legacy pipeline emits a fold's trip test and step beside the scope of
/// its parent, so a stack reload there (a binder held in a slot) is counted in
/// the parent's trips and not the fold's; its pool reads are moved to the fold,
/// where they run. That undercounts legacy's loads, never overcounts them.
const LEGACY_AVX2: [Cost; 11] = [
    Cost {
        bytes: 724,
        loads: 33,
        stores: 18,
        pool_reads: 49,
    },
    Cost {
        bytes: 728,
        loads: 33,
        stores: 18,
        pool_reads: 49,
    },
    Cost {
        bytes: 1056,
        loads: 89,
        stores: 26,
        pool_reads: 196,
    },
    Cost {
        bytes: 1056,
        loads: 71,
        stores: 36,
        pool_reads: 358,
    },
    Cost {
        bytes: 247328,
        loads: 195968,
        stores: 13947,
        pool_reads: 2177,
    },
    Cost {
        bytes: 3012,
        loads: 1769,
        stores: 189,
        pool_reads: 190,
    },
    Cost {
        bytes: 1136,
        loads: 68,
        stores: 7,
        pool_reads: 48,
    },
    Cost {
        bytes: 932,
        loads: 6,
        stores: 2,
        pool_reads: 43,
    },
    Cost {
        bytes: 636,
        loads: 3,
        stores: 1,
        pool_reads: 43,
    },
    Cost {
        bytes: 500,
        loads: 0,
        stores: 0,
        pool_reads: 39,
    },
    Cost {
        bytes: 420428,
        loads: 165548,
        stores: 62065,
        pool_reads: 131,
    },
];

fn print_table(measured: &[Cost]) -> String {
    measured
        .iter()
        .map(|c| {
            format!(
                "    Cost {{ bytes: {}, loads: {}, stores: {}, pool_reads: {} }},",
                c.bytes, c.loads, c.stores, c.pool_reads
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn legacy_traffic_is_the_recorded_table() {
    if selection() || jit_vector_bytes() != AVX2_VECTOR_BYTES {
        return;
    }
    let measured = costs();
    assert_eq!(
        measured,
        LEGACY_AVX2,
        "the legacy pipeline's traffic moved; if the metric changed on purpose, \
         re-record LEGACY_AVX2:\n{}",
        print_table(&measured)
    );
}

/// Per row, at most a quarter more memory operations than legacy and a few
/// more, in all at most legacy's, and no more of the stack's and no more code.
///
/// The slack is for the rows whose traffic is a handful of constants: one
/// pool read more there is a large fraction. The stack traffic and the code
/// have none, because the new pipeline is below the old on every row today.
#[test]
fn selection_moves_no_more_memory_than_legacy() {
    if !selection() {
        return;
    }
    const SLACK: u64 = 8;
    let measured = costs();
    let mut worse = Vec::new();
    for ((row, now), then) in rows::TABLE.iter().zip(&measured).zip(LEGACY_AVX2) {
        let (now_memory, then_memory) = (now.memory(), then.memory());
        let stack = (now.loads + now.stores, then.loads + then.stores);
        if now_memory * 4 > then_memory * 5 + 4 * SLACK
            || stack.0 > stack.1
            || now.bytes > then.bytes
        {
            worse.push(format!(
                "{}: memory {now_memory} against {then_memory}, stack {} against {}, \
                 code {} against {} bytes",
                row.name, stack.0, stack.1, now.bytes, then.bytes
            ));
        }
    }
    assert!(
        worse.is_empty(),
        "worse than legacy by the ratchet's bound:\n{}",
        worse.join("\n")
    );
    let (now, then): (u64, u64) = (
        measured.iter().copied().map(Cost::memory).sum(),
        LEGACY_AVX2.iter().copied().map(Cost::memory).sum(),
    );
    assert!(
        now <= then,
        "over every row, {now} memory operations against legacy's {then}"
    );
}
