//! A program too big for one e-graph compiles, a unit at a time.
//!
//! docs/plans/2026-09-25-the-language-is-kernel.md §4, O1: Noto's ASCII
//! inlined is 169,263 classes before any rule fires, over the e-graph's hard
//! class limit, so its saturation stops at once and the whole font is
//! emitted as written. Each glyph is a *unit* (`Kernel::by_ref`): saturated and extracted
//! by itself, held in the program as an opaque leaf, and linked back in after
//! extraction. This builds a program of that kind — units over fresh
//! uniforms, under an `if id < k` tree, more classes inlined than the limit —
//! and pins that it compiles, that every saturation it pays is a unit's or the
//! tree's and runs under its own cap, and that every id draws its unit.
//!
//! Its own binary, because it points the process-global telemetry sink at a
//! file and reads every record the process writes.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_core::{Kernel, Lattice, Manifold, Uniform};
use pixelflow_ir::OpKind;
use pixelflow_search::egraph::{EGraph, HARD_CLASS_LIMIT, Vocabulary, insert};

/// How many units the program holds.
const UNITS: usize = 64;

/// Pairs of redundant negations each unit is padded with: enough that the
/// units inlined pass [`HARD_CLASS_LIMIT`] (64 units of about 1,700 nodes),
/// and all of it gone after one rule and congruence, so each unit saturates
/// in two rounds and the linked program is small enough to emit. (A chain of
/// `+ 0` does not: associativity and commutativity grow it to the class cap,
/// a saturation of minutes in the debug profile.)
const PADDING: usize = 850;

/// The frame every id is drawn over.
const FRAME: usize = 4;

/// A unit's own value at its sample, written plainly: what the padded unit
/// denotes, and what each id must draw.
fn plain(index: usize, scale: &Kernel) -> Kernel {
    // Two structures, alternating: two unit saturations, not one per unit.
    match index % 2 {
        0 => Kernel::x().mul(scale),
        _ => Kernel::y().mul(scale).add(&Kernel::x()),
    }
}

/// [`plain`] buried under [`PADDING`] pairs of negations, pushed onto one
/// arena (composing them as kernels would copy the chain at every level).
fn padded(index: usize, scale: &Kernel) -> Kernel {
    let unpadded = plain(index, scale);
    let (arena, root) = unpadded.parts();
    let mut arena = arena.clone();
    let root = (0..PADDING).fold(root, |acc, _| {
        let once = arena.push_unary(OpKind::Neg, acc);
        arena.push_unary(OpKind::Neg, once)
    });
    Kernel::from_parts(arena, root)
}

/// The value unit `index` draws at `(x, y)`, in host arithmetic.
fn expected(index: usize, scale: f32, x: f32, y: f32) -> f32 {
    match index % 2 {
        0 => x * scale,
        _ => y * scale + x,
    }
}

/// Each unit's scale, a uniform with this default: exact in `f32` times a
/// small integer, so the host arithmetic is the kernel's to the bit.
fn scale_of(index: usize) -> f32 {
    index as f32 + 0.25
}

/// `units` under a balanced tree of `if id < k`, the font's id tree.
fn id_tree(id: &Kernel, units: &[Kernel], first: usize) -> Kernel {
    if units.len() == 1 {
        return units[0].clone();
    }
    let half = units.len() / 2;
    let k = Kernel::constant((first + half) as f32);
    id.lt(&k).select(
        &id_tree(id, &units[..half], first),
        &id_tree(id, &units[half..], first + half),
    )
}

/// One telemetry record's integer field.
fn field(record: &str, name: &str) -> u64 {
    let start = record
        .find(&format!("\"{name}\":"))
        .unwrap_or_else(|| panic!("no {name} in {record}"))
        + name.len()
        + 3;
    record[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .unwrap_or_else(|e| panic!("{name} in {record}: {e}"))
}

#[test]
fn a_program_past_the_class_limit_compiles_unit_by_unit() {
    let sink = std::env::temp_dir().join(format!(
        "pixelflow-units-telemetry-{}.jsonl",
        std::process::id()
    ));
    // Removed first: a stale file from a reused pid would be read as this
    // run's records. Absent is the expected case.
    if sink.exists() {
        std::fs::remove_file(&sink).expect("clear the telemetry sink");
    }
    // SAFETY: this binary has one test, and nothing else in it reads the
    // environment concurrently.
    unsafe { std::env::set_var("PIXELFLOW_SATURATION_TELEMETRY", &sink) };

    let id = Uniform::new(0.0);
    let scales: Vec<Uniform> = (0..UNITS).map(|i| Uniform::new(scale_of(i))).collect();
    let units: Vec<Kernel> = scales
        .iter()
        .enumerate()
        .map(|(i, s)| padded(i, &s.kernel()).by_ref())
        .collect();
    let program = id_tree(&id.kernel(), &units, 0);

    // Inlined, the program is past the e-graph's hard limit before any rule
    // fires: one saturation could not have optimized it.
    let (inlined, inlined_root) = program.linked_parts();
    let mut graph = EGraph::new();
    let _root_class = insert(&inlined, inlined_root, &mut graph, Vocabulary::Runtime)
        .expect("the inlined program inserts");
    assert!(
        graph.num_classes() > HARD_CLASS_LIMIT,
        "the fixture must be past the limit inlined: {} classes",
        graph.num_classes()
    );

    let extent = u32::try_from(FRAME).expect("FRAME fits a u32");
    let manifold = Manifold::compile(&program, [extent, extent]);

    let records = std::fs::read_to_string(&sink).expect("the telemetry sink was written");
    std::fs::remove_file(&sink).expect("remove the telemetry sink");
    let records: Vec<&str> = records.lines().collect();
    assert!(
        records.iter().all(|r| !r.contains("\"declined\"")),
        "nothing here is outside the e-graph's vocabulary: {records:#?}"
    );
    assert_eq!(
        records.len(),
        3,
        "one saturation per unit structure (two) and one for the id tree: {records:#?}"
    );
    for record in &records {
        assert!(
            field(record, "inserted_classes") < field(record, "max_classes"),
            "a unit inserts under its own cap: {record}"
        );
        assert!(
            field(record, "iterations") >= 1,
            "and rules fire on it: {record}"
        );
    }

    let mut block = manifold.block();
    for unit in 0..UNITS {
        block
            .set(id, unit as f32)
            .expect("id is the program's argument");
        let bound = manifold.bind(&[]).with_uniforms(&block);
        let frame = Lattice::frame(FRAME, FRAME).collapse(&bound);
        let drawn = frame.buffer();
        for row in 0..FRAME {
            for col in 0..FRAME {
                let want = expected(unit, scale_of(unit), col as f32, row as f32);
                assert_eq!(
                    drawn[row * FRAME + col].to_bits(),
                    want.to_bits(),
                    "id {unit} at ({col}, {row}) drew {} for {want}",
                    drawn[row * FRAME + col]
                );
            }
        }
    }
}
