//! The kernels the harness measures, and the fixture format that stores them.
//!
//! A cost model is fitted to *what production compiles*, so the corpus is led
//! by the glyph bakes `core-term` runs (`GlyphAtlas::warm`, one fused kernel
//! per printable character at each display density) and filled out by three
//! synthetic families that isolate the structures the allocator trades
//! against. Each entry carries the **shape** it is baked at, because that is
//! what turns a static count into a dynamic one — the omission that made the
//! previous allocator measurements unable to see their own units
//! (`docs/plans/2026-09-01-register-allocation-escape-hatches.md`, 3″).
//!
//! Capture writes the corpus once, to files; the bench replays those files.
//! So the corpus is a fixture with a diff, not a side effect of whichever
//! test suite happened to run, and every allocation variant is measured on
//! byte-identical input.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{BufferDecl, BufferId, BufferIdentity, ExprArena, ExprId, ExprNode};
use pixelflow_ir::fold::Fold;

/// One kernel and the lattice it is baked at.
pub struct CollapseKernel {
    /// Unique within a corpus; the row key in the output.
    pub name: String,
    /// Which family it came from — the grouping the analysis reports by.
    pub family: String,
    pub arena: ExprArena,
    pub root: ExprId,
    /// The lattice extent, exactly as `Lattice::bake` would see it.
    pub extent: [u32; 2],
    /// Captured contents for each buffer `arena` declares, aligned by
    /// [`BufferId`]: `arena.buffers()[i]` is slot `i`'s declaration,
    /// `buffer_data[i]` is what production actually bound there. `None` at a
    /// slot means capture had nothing real for it.
    ///
    /// This is not cosmetic: collapse cost is *not* independent of a
    /// buffer's values. `emit_skip_if_all_false`/`emit_skip_if_all_true`
    /// (`pixelflow-codegen/src/emit/mod.rs`) branch on a `Select` guard's
    /// mask at runtime, and a zero-filled piece table makes a glyph's every
    /// crossing-span mask uniformly false — the guard skips an arm
    /// production always takes. Replaying zeros here measures that skipped
    /// arm's absence, not its cost. The replay path
    /// (`collapse_bench::dummy_context`) binds `Some` data verbatim and
    /// falls back to zeros only for a genuinely uncaptured slot, loudly.
    pub buffer_data: Vec<Option<Arc<Vec<f32>>>>,
}

/// How many times each scope of the collapse nest runs, for one
/// `call_collapse` at this extent and vector width.
///
/// The collapse ABI's own arithmetic: the frame prologue runs once, the row
/// prologue once per row, the body once per full SIMD group per row. The
/// scalar tail `Lattice::bake` walks afterwards is not this kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trips {
    pub rows: u64,
    pub groups: u64,
}

impl Trips {
    /// # Panics
    /// If the extent is narrower than one SIMD group — the collapse kernel is
    /// never called for such a lattice, so there would be nothing to time.
    #[must_use]
    pub fn of(extent: [u32; 2], lanes: u32) -> Self {
        assert!(
            extent[0] >= lanes,
            "extent {extent:?} is narrower than the {lanes}-lane batch: bake would run \
             the scalar tail only and never call the collapse kernel"
        );
        Self {
            rows: u64::from(extent[1]),
            groups: u64::from(extent[0] / lanes),
        }
    }
}

// =============================================================================
// Fixture format
// =============================================================================

// v2 adds `A <default-bits>`, a kernel argument. v1 had no such node
// because the `invariant` family's frame-scope leaf was the Z axis, and a
// lattice has two axes now — so an argument is the only leaf that is
// invariant across the lattice and survives constant folding. A v1 fixture
// cannot be replayed into a v2 corpus and the header says so rather than
// letting the ids drift silently.
//
// v3 adds `B <id> <width> <height>` (a declared buffer slot) and
// `N <op> <child>...` (an n-ary node — currently only `Reduce`, the winding
// fold). Both were rejected in v2 on the premise that a corpus kernel
// "reading a bound buffer would be unlike anything production bakes" —
// false since a glyph's winding became a `Kernel::sum_over` over a bound
// piece table (`pixelflow-graphics`'s `loop_blinn::glyph`), which is exactly
// this shape. `Param` is still rejected: nothing production bakes carries an
// unsubstituted macro parameter.
//
// v3 carried a buffer's *shape* only; replay bound every declared buffer to
// zeros (`dummy_context`) on the premise that "collapse cost depends on the
// arena's shape, not the buffer's values." That premise was false: a
// `Select` guard's runtime skip (`emit_skip_if_all_false`/`_all_true` in
// `pixelflow-codegen/src/emit/mod.rs`) branches on whether any lane's mask
// is set, which is a fact about the *data*, not the shape. A zero-filled
// piece table makes every one of a glyph's crossing-span masks uniformly
// false, so v3 measured a control-flow path production never takes. v4 adds
// `D <slot> <bits>...` — the slot's captured contents, as bit patterns like
// every other value this format stores, so replay executes the same guard
// decisions production does.
const HEADER: &str = "# pixelflow collapse corpus v4";
/// Versions this format replaced, recognised only so [`decode`] can say
/// *which* mismatch it hit and why regenerating is the fix.
const SUPERSEDED_HEADERS: &[(&str, &str)] = &[
    (
        "# pixelflow collapse corpus v1",
        "v1 had no way to spell an argument node (`A`)",
    ),
    (
        "# pixelflow collapse corpus v2",
        "v2 had no way to spell a declared buffer (`B`) or an n-ary node (`N`)",
    ),
    (
        "# pixelflow collapse corpus v3",
        "v3 declared a buffer's shape but not its contents, so replay bound zeros \
         and never exercised the guard skip a real mask fires",
    ),
];

/// Write `kernels` into `dir`, one `.collapse` file each.
///
/// The node encoding is the arena dumpers' (`pixelflow-core`'s cell-grid
/// dumper, `pixelflow-graphics`'s glyph dumper): reachable nodes in ascending
/// id order with ids remapped dense, constants as bit patterns. The additions
/// are the `family` and `extent` lines — the shape, which is the point — plus
/// `B`/`N` for a declared buffer slot and an n-ary node, which the dumpers
/// this format borrows from don't need because they always hold a bakeable
/// kernel with its buffers already bound to real memory.
///
/// # Panics
/// If the directory cannot be created, a file cannot be written, or a kernel
/// holds a `Param` node — a macro front-end placeholder that never survives
/// to a compiled kernel, so a corpus entry carrying one is a corpus bug, not
/// a shape production bakes.
pub fn write_dir(dir: &Path, kernels: &[CollapseKernel]) {
    std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    for kernel in kernels {
        let path = dir.join(format!("{}.collapse", kernel.name));
        std::fs::write(&path, encode(kernel))
            .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }
}

/// Read every `.collapse` file in `dir`, in name order.
///
/// # Panics
/// If the directory cannot be read or any file is not a current-version
/// fixture (see [`HEADER`]).
#[must_use]
pub fn read_dir(dir: &Path) -> Vec<CollapseKernel> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "collapse"))
        .collect();
    paths.sort();
    paths.iter().map(|p| decode(p)).collect()
}

/// The fixture text for one kernel. Also the round-trip oracle: two kernels
/// with the same encoding are the same kernel.
#[must_use]
pub fn encode(kernel: &CollapseKernel) -> String {
    use std::fmt::Write as _;

    let (arena, root) = (&kernel.arena, kernel.root);
    let len = arena.nodes_raw().len();
    let mut reachable = vec![false; len];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut reachable[id.0 as usize], true) {
            continue;
        }
        stack.extend(arena.children(id));
    }

    let mut out = String::new();
    writeln!(out, "{HEADER}").expect("fmt");
    writeln!(out, "name {}", kernel.name).expect("fmt");
    writeln!(out, "family {}", kernel.family).expect("fmt");
    let [ex, ey] = kernel.extent;
    writeln!(out, "extent {ex} {ey}").expect("fmt");

    let mut dense: Vec<u32> = vec![u32::MAX; len];
    let mut next = 0u32;
    let d = |dense: &[u32], id: ExprId| -> u32 {
        let v = dense[id.0 as usize];
        assert_ne!(v, u32::MAX, "child dumped before parent");
        v
    };
    // A buffer with `n` gathers into it dumps `n` `B` lines (one per
    // `Buffer` leaf), all naming the same slot — see the comment on that
    // arm below. Its `D` line is data, not shape, so it must not repeat
    // `n` times too; this tracks which slots already got theirs.
    let mut buffer_data_emitted: HashSet<u16> = HashSet::new();
    for idx in 0..len {
        if !reachable[idx] {
            continue;
        }
        let id = ExprId(idx as u32);
        match arena.node(id) {
            ExprNode::Var(i) => writeln!(out, "V {i}"),
            // A kernel argument. The format knows about one because, with
            // two coordinate axes, a `Uniform` is the *only* leaf that is
            // both invariant across the lattice and beyond the constant
            // folder's reach — which is precisely what the `invariant`
            // family needs to give LICM's frame prologue something to lift.
            // The Z axis used to serve that role; it was the same thing
            // wearing a coordinate's name.
            ExprNode::Uniform(u) => {
                writeln!(out, "A {}", arena.uniform_decl(*u).default.to_bits())
            }
            ExprNode::Const(v) => writeln!(out, "C {}", v.to_bits()),
            // A declared buffer slot. `id.0` is the *arena's* slot index, not
            // a [`BufferIdentity`] — identities are minted and mean nothing
            // across a decode, but the slot index is what lets several
            // `Buffer` leaves (one per `Kernel::at` gather into the same
            // table) fold back onto one declared slot instead of each
            // minting its own on decode, which would change the arena's
            // shape (`ExprArena::buffers().len()`, and with it every
            // `Uniform`'s context slot).
            ExprNode::Buffer(id) => {
                let decl = arena.buffer_decl(*id);
                writeln!(out, "B {} {} {}", id.0, decl.width, decl.height).expect("fmt");
                // Only the first occurrence of this slot writes its data —
                // see `buffer_data_emitted` above.
                if buffer_data_emitted.insert(id.0) {
                    write_buffer_data(&mut out, kernel, *id);
                }
                Ok(())
            }
            ExprNode::Unary(k, a) => writeln!(out, "U {k:?} {}", d(&dense, *a)),
            ExprNode::Binary(k, a, b) => {
                writeln!(out, "Bi {k:?} {} {}", d(&dense, *a), d(&dense, *b))
            }
            ExprNode::Ternary(k, a, b, c) => writeln!(
                out,
                "T {k:?} {} {} {}",
                d(&dense, *a),
                d(&dense, *b),
                d(&dense, *c)
            ),
            // An n-ary node — in practice `Reduce`, the winding fold's
            // binder: `[Const(combiner), Const(reduce_var), Const(extent),
            // body]`. The three `Const` children round trip through the `C`
            // arm above like any other constant; this arm only has to spell
            // the child list itself, whatever its length.
            ExprNode::Nary(k, start, count) => {
                let children = arena.nary_children_slice(*start, *count);
                let ids: Vec<u32> = children.iter().map(|c| d(&dense, *c)).collect();
                write!(out, "N {k:?}").expect("fmt");
                for id in ids {
                    write!(out, " {id}").expect("fmt");
                }
                writeln!(out)
            }
            // The fold as opaque bits — it is metadata, not children, so it
            // travels as one field rather than as three serialized nodes.
            ExprNode::Reduce { fold, body } => {
                writeln!(out, "R {} {}", fold.to_bits(), d(&dense, *body))
            }
            ExprNode::Param(i) => panic!(
                "{}: corpus kernels must be bakeable, but this one holds Param({i}) — a \
                 macro front-end placeholder, never present in a compiled kernel",
                kernel.name
            ),
            // A key names an entry in this process's `KernelStore`, which a
            // corpus file outlives; expand references before dumping one.
            ExprNode::Ref(k) => panic!(
                "{}: corpus kernels must be self-contained, but this one holds Ref({k:?}) — \
                 a name for a kernel interned in this process only",
                kernel.name
            ),
        }
        .expect("fmt");
        dense[idx] = next;
        next += 1;
    }
    writeln!(out, "root {}", d(&dense, root)).expect("fmt");
    out
}

/// Write buffer `slot`'s captured contents as a `D` line — one bit pattern
/// per `f32`, the way the format already stores every other value, because a
/// decimal round trip is not exact and this data decides guard masks. A slot
/// `kernel` has no captured contents for is not written at all: the replay
/// path's zero fallback (`dummy_context`) is the loud, explicit case, not a
/// silent default baked into the fixture.
///
/// # Panics
/// If writing to `out` fails (an allocation failure, effectively
/// unreachable for a `String` target).
fn write_buffer_data(out: &mut String, kernel: &CollapseKernel, slot: BufferId) {
    use std::fmt::Write as _;
    let Some(data) = kernel
        .buffer_data
        .get(slot.0 as usize)
        .and_then(Option::as_ref)
    else {
        return;
    };
    write!(out, "D {}", slot.0).expect("fmt");
    for v in data.iter() {
        write!(out, " {}", v.to_bits()).expect("fmt");
    }
    writeln!(out).expect("fmt");
}

fn decode(path: &Path) -> CollapseKernel {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut lines = text.lines();
    // Name the version mismatch here rather than letting a stale body surface
    // as "unparseable line": a stale corpus is regenerated, not debugged.
    match lines.next() {
        Some(HEADER) => {}
        Some(found) => match SUPERSEDED_HEADERS.iter().find(|(h, _)| *h == found) {
            Some((_, why)) => panic!(
                "{}: this is a {found:?} corpus and the format is now {HEADER:?} — {why}. \
                 Regenerate the corpus; a stale fixture cannot be replayed into this arena.",
                path.display()
            ),
            None => panic!(
                "{}: not a collapse corpus fixture (expected {HEADER:?}, found {found:?})",
                path.display()
            ),
        },
        None => panic!("{}: empty fixture (expected {HEADER:?})", path.display()),
    }

    let mut name = None;
    let mut family = None;
    let mut extent = None;
    let mut root = None;
    let mut arena = ExprArena::new();
    let mut next_id = 0u32;
    // Original arena slot index -> the slot this decode declared for it. A
    // buffer with `n` gathers into it dumps `n` separate `B` lines (one per
    // `Buffer` leaf — the arena has no hash-consing), all naming the same
    // original slot; the first declares it here, the rest must fold onto
    // that same declaration or the decoded arena would gain buffer slots the
    // original never had, shifting every `Uniform`'s context slot
    // (`ExprArena::buffers().len()`).
    let mut buffer_slots: HashMap<u16, BufferId> = HashMap::new();
    // Decoded slot -> its captured contents, from that slot's `D` line (at
    // most one, written once per slot regardless of how many `B` lines name
    // it — see `write_buffer_data`). Absent for a slot capture had nothing
    // for; folded into `CollapseKernel::buffer_data` once every buffer is
    // declared, below.
    let mut buffer_data: HashMap<BufferId, Arc<Vec<f32>>> = HashMap::new();

    let op = |s: &str| -> OpKind {
        OpKind::all()
            .find(|k| format!("{k:?}") == s)
            .unwrap_or_else(|| panic!("{}: unknown OpKind {s:?}", path.display()))
    };
    let id = |s: &str| -> ExprId {
        ExprId(
            s.parse()
                .unwrap_or_else(|e| panic!("{}: bad id {s:?}: {e}", path.display())),
        )
    };
    let dim = |s: &str| -> u32 {
        s.parse()
            .unwrap_or_else(|e| panic!("{}: bad extent {s:?}: {e}", path.display()))
    };

    for line in lines {
        let f: Vec<&str> = line.split_whitespace().collect();
        let pushed = match f.as_slice() {
            ["name", n] => {
                name = Some((*n).to_string());
                continue;
            }
            ["family", n] => {
                family = Some((*n).to_string());
                continue;
            }
            ["extent", x, y] => {
                extent = Some([dim(x), dim(y)]);
                continue;
            }
            ["root", r] => {
                root = Some(id(r));
                continue;
            }
            ["V", i] => arena.push_var(i.parse().expect("var index")),
            ["A", bits] => {
                let default = f32::from_bits(bits.parse().expect("argument default bits"));
                let slot = arena.declare_uniform(pixelflow_ir::Uniform::new(default).decl());
                arena.push_uniform(slot)
            }
            ["C", bits] => arena.push_const(f32::from_bits(bits.parse().expect("const bits"))),
            ["B", orig_slot, w, h] => {
                let orig_slot: u16 = orig_slot.parse().unwrap_or_else(|e| {
                    panic!("{}: bad buffer slot {orig_slot:?}: {e}", path.display())
                });
                let (width, height) = (dim(w), dim(h));
                let slot = *buffer_slots.entry(orig_slot).or_insert_with(|| {
                    arena.declare_buffer(BufferDecl {
                        id: BufferIdentity::mint(),
                        width,
                        height,
                    })
                });
                let decl = arena.buffer_decl(slot);
                assert_eq!(
                    (decl.width, decl.height),
                    (width, height),
                    "{}: buffer slot {orig_slot} redeclared at a different shape",
                    path.display()
                );
                arena.push_buffer(slot)
            }
            ["D", orig_slot, bits @ ..] => {
                let orig_slot: u16 = orig_slot.parse().unwrap_or_else(|e| {
                    panic!(
                        "{}: bad buffer data slot {orig_slot:?}: {e}",
                        path.display()
                    )
                });
                let slot = *buffer_slots.get(&orig_slot).unwrap_or_else(|| {
                    panic!(
                        "{}: D line for buffer slot {orig_slot} appeared before its B line \
                         declared it",
                        path.display()
                    )
                });
                let decl = arena.buffer_decl(slot);
                let expected = decl.width as usize * decl.height as usize;
                assert_eq!(
                    bits.len(),
                    expected,
                    "{}: buffer slot {orig_slot} data has {} value(s), the declared {}x{} \
                     extent wants {expected}",
                    path.display(),
                    bits.len(),
                    decl.width,
                    decl.height
                );
                let data: Vec<f32> = bits
                    .iter()
                    .map(|b| {
                        f32::from_bits(b.parse().unwrap_or_else(|e| {
                            panic!("{}: bad buffer data bits {b:?}: {e}", path.display())
                        }))
                    })
                    .collect();
                buffer_data.insert(slot, Arc::new(data));
                continue;
            }
            ["U", k, a] => arena.push_unary(op(k), id(a)),
            ["Bi", k, a, b] => arena.push_binary(op(k), id(a), id(b)),
            ["T", k, a, b, c] => arena.push_ternary(op(k), id(a), id(b), id(c)),
            ["N", k, children @ ..] => {
                let children: Vec<ExprId> = children.iter().map(|c| id(c)).collect();
                arena.push_nary(op(k), &children)
            }
            ["R", bits, body] => {
                let bits: u64 = bits
                    .parse()
                    .unwrap_or_else(|e| panic!("{}: bad fold bits {bits:?}: {e}", path.display()));
                let fold = Fold::from_bits(bits)
                    .unwrap_or_else(|| panic!("{}: {bits} names no fold", path.display()));
                arena.push_reduce(fold, id(body))
            }
            other => panic!("{}: unparseable line {other:?}", path.display()),
        };
        assert_eq!(
            pushed,
            ExprId(next_id),
            "{}: replay drifted from dumped ids",
            path.display()
        );
        next_id += 1;
    }

    // Aligned by `BufferId` — `arena.buffers()[i]` is slot `i`'s
    // declaration — so a slot with no `D` line decodes to `None`, exactly
    // the shape `dummy_context` expects for a genuinely uncaptured slot.
    let buffer_data_by_slot: Vec<Option<Arc<Vec<f32>>>> = (0..arena.buffers().len())
        .map(|i| buffer_data.get(&BufferId(i as u16)).cloned())
        .collect();

    CollapseKernel {
        name: name.unwrap_or_else(|| panic!("{}: no name", path.display())),
        family: family.unwrap_or_else(|| panic!("{}: no family", path.display())),
        arena,
        root: root.unwrap_or_else(|| panic!("{}: no root", path.display())),
        extent: extent.unwrap_or_else(|| panic!("{}: no extent", path.display())),
        buffer_data: buffer_data_by_slot,
    }
}

// =============================================================================
// The synthetic families
// =============================================================================

/// Shape the pressure families are baked at: wide enough that the body
/// dominates, small enough to keep a sample cheap.
const PRESSURE_EXTENT: [u32; 2] = [256, 64];
/// Where a loop-invariant term is amortized over many body iterations.
const INVARIANT_HOT_EXTENT: [u32; 2] = [256, 256];
/// Where it is not: two rows of a few batch groups pay the prologues in full.
const INVARIANT_COLD_EXTENT: [u32; 2] = [64, 2];
/// The value the `invariant` family's frame-scope argument carries. Any
/// finite number does — the timing does not depend on it — but the block the
/// runner passes must agree with this so the kernel reads a real `f32`.
pub const CORPUS_ARG: f32 = 1.0;

/// Every synthetic kernel, in a fixed order.
///
/// Three families, each isolating one thing the allocator trades:
/// - `wide{n}` — a balanced tree of `n` leaves: transient pressure only, no
///   loop-invariant structure, so the prologues stay empty and the whole cost
///   is the body's;
/// - `anchored{w}x{d}` — `w` values computed up front and folded in after a
///   chain of depth `d`, so `w` live ranges cross `d` instructions: the shape
///   that makes eviction choose;
/// - `invariant{n}` — `n` X-invariant terms each read once by the body, at a
///   hot and a cold shape: the same static code with the trip count changed,
///   which is exactly the axis a static memory-op count cannot see.
#[must_use]
pub fn synthetic() -> Vec<CollapseKernel> {
    let mut out = Vec::new();
    let mut push =
        |name: String, family: &str, extent: [u32; 2], build: &dyn Fn(&mut ExprArena) -> ExprId| {
            let mut arena = ExprArena::new();
            let root = build(&mut arena);
            out.push(CollapseKernel {
                name,
                family: family.to_string(),
                arena,
                root,
                extent,
                // None of the synthetic families declare a buffer.
                buffer_data: Vec::new(),
            });
        };
    for n in [8usize, 16, 32, 64] {
        push(
            format!("wide{n:03}"),
            "wide",
            PRESSURE_EXTENT,
            &move |a: &mut ExprArena| wide(a, n),
        );
    }
    for (w, d) in [(8usize, 24usize), (12, 40), (16, 64)] {
        push(
            format!("anchored{w:02}x{d:02}"),
            "anchored",
            PRESSURE_EXTENT,
            &move |a: &mut ExprArena| anchored(a, w, d),
        );
    }
    for n in [4usize, 8, 16, 48] {
        for (tag, extent) in [
            ("hot", INVARIANT_HOT_EXTENT),
            ("cold", INVARIANT_COLD_EXTENT),
        ] {
            push(
                format!("invariant{n:02}_{tag}"),
                &format!("invariant_{tag}"),
                extent,
                &move |a: &mut ExprArena| invariants(a, n),
            );
        }
    }
    out
}

/// A leaf that varies in X, salted so the tree is not one common
/// subexpression the optimizer folds away.
fn x_leaf(a: &mut ExprArena, salt: usize) -> ExprId {
    let x = a.push_var(0);
    let c = a.push_const(0.125 + (salt % 13) as f32 * 0.0625);
    a.push_binary(OpKind::Mul, x, c)
}

/// A balanced Add/Sub tree over `n` X-varying leaves.
fn wide(a: &mut ExprArena, n: usize) -> ExprId {
    assert!(n.is_power_of_two(), "wide takes a power of two, got {n}");
    let mut level: Vec<ExprId> = (0..n).map(|i| x_leaf(a, i)).collect();
    let mut salt = 0usize;
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|pair| {
                salt += 1;
                let op = if salt.is_multiple_of(3) {
                    OpKind::Sub
                } else {
                    OpKind::Add
                };
                a.push_binary(op, pair[0], pair[1])
            })
            .collect();
    }
    level[0]
}

/// `w` anchors computed first, a dependent chain of depth `d`, then the
/// anchors folded in — so every anchor is live across the whole chain.
fn anchored(a: &mut ExprArena, w: usize, d: usize) -> ExprId {
    let anchors: Vec<ExprId> = (0..w)
        .map(|i| {
            let leaf = x_leaf(a, i * 7 + 1);
            a.push_unary(OpKind::Sqrt, leaf)
        })
        .collect();
    let mut chain = x_leaf(a, 991);
    for i in 0..d {
        let c = a.push_const(1.0 + (i % 5) as f32 * 0.25);
        chain = a.push_ternary(OpKind::MulAdd, chain, c, chain);
    }
    anchors.iter().fold(chain, |acc, &anchor| {
        a.push_binary(OpKind::Add, acc, anchor)
    })
}

/// `n` terms invariant in X — half of them invariant in Y as well, so both
/// prologues get work — each read exactly once by an X-varying body term.
fn invariants(a: &mut ExprArena, n: usize) -> ExprId {
    let y = a.push_var(1);
    // Frame scope needs a leaf the folder cannot collapse and the lattice
    // cannot vary. That is a kernel argument; it used to be the Z axis,
    // which was the same thing wearing a coordinate's name. A `Const` would
    // fold and leave LICM nothing to lift.
    let arg = a.declare_uniform(pixelflow_ir::Uniform::new(CORPUS_ARG).decl());
    let z = a.push_uniform(arg);
    let terms: Vec<ExprId> = (0..n)
        .map(|i| {
            let c = a.push_const(0.5 + i as f32 * 0.125);
            let base = if i.is_multiple_of(2) {
                // Frame scope: reads neither X nor Y.
                a.push_binary(OpKind::Mul, z, c)
            } else {
                // Row scope: reads Y.
                let scaled = a.push_binary(OpKind::Mul, y, c);
                a.push_binary(OpKind::Add, scaled, z)
            };
            let one = a.push_const(1.0);
            let positive = a.push_binary(OpKind::Add, base, one);
            a.push_unary(OpKind::Sqrt, positive)
        })
        .collect();
    let x = a.push_var(0);
    terms.iter().enumerate().fold(x, |acc, (i, &term)| {
        let leaf = x_leaf(a, i * 3 + 5);
        let scaled = a.push_binary(OpKind::Mul, leaf, term);
        a.push_binary(OpKind::Add, acc, scaled)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_synthetic_kernel_has_a_unique_name_and_a_bakeable_shape() {
        let kernels = synthetic();
        let mut names: Vec<&str> = kernels.iter().map(|k| k.name.as_str()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate kernel name in the corpus");
        for kernel in &kernels {
            // 16 lanes is the widest tier this repo emits; a corpus entry that
            // cannot fill one group there is one the bench would skip.
            let trips = Trips::of(kernel.extent, 16);
            assert!(trips.rows > 0 && trips.groups > 0, "{}", kernel.name);
        }
    }

    #[test]
    fn a_written_corpus_reads_back_identical() {
        let dir =
            std::env::temp_dir().join(format!("pixelflow-collapse-corpus-{}", std::process::id()));
        let kernels = synthetic();
        write_dir(&dir, &kernels);
        let back = read_dir(&dir);
        assert_eq!(back.len(), kernels.len());
        let mut sorted: Vec<&CollapseKernel> = kernels.iter().collect();
        sorted.sort_by(|a, b| a.name.cmp(&b.name));
        for (want, got) in sorted.iter().zip(&back) {
            assert_eq!(
                encode(want),
                encode(got),
                "{}: the fixture did not round trip",
                want.name
            );
        }
        std::fs::remove_dir_all(&dir).expect("clean up");
    }

    /// A stale v1 corpus must say so. Nothing commits corpora, so failing is
    /// right — but "unparseable line" would send the reader into their file
    /// instead of into `gen_bench_corpus`.
    #[test]
    #[should_panic(expected = "v1 had no way to spell an argument node")]
    fn a_v1_corpus_names_the_version_it_is() {
        let dir = std::env::temp_dir().join(format!(
            "pixelflow-collapse-corpus-v1-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("stale.txt");
        // Index 0 is v1, per `SUPERSEDED_HEADERS`'s declaration order.
        let (header, _) = SUPERSEDED_HEADERS[0];
        std::fs::write(
            &path,
            format!("{header}\nname stale\nfamily wide\nextent 8 8\nV 0\nroot 0\n"),
        )
        .expect("write");
        let _ = decode(&path);
    }

    /// A stale v2 corpus — the format before buffers and n-ary nodes — must
    /// say so too, not just v1.
    #[test]
    #[should_panic(expected = "v2 had no way to spell a declared buffer")]
    fn a_v2_corpus_names_the_version_it_is() {
        let dir = std::env::temp_dir().join(format!(
            "pixelflow-collapse-corpus-v2-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("stale.txt");
        // Index 1 is v2, per `SUPERSEDED_HEADERS`'s declaration order.
        let (header, _) = SUPERSEDED_HEADERS[1];
        std::fs::write(
            &path,
            format!("{header}\nname stale\nfamily wide\nextent 8 8\nV 0\nroot 0\n"),
        )
        .expect("write");
        let _ = decode(&path);
    }

    /// A kernel that declares a buffer round trips through the fixture text
    /// exactly, and the decoded arena is bakeable end to end: compile it at
    /// its own shape, bind a zero-filled buffer of the declared extent (the
    /// corpus never carries real pixel data — collapse cost is a function of
    /// the arena's shape, not what a gather reads), and collapse one call.
    #[test]
    fn a_buffer_declaring_kernel_round_trips_and_bakes() {
        let mut arena = ExprArena::new();
        let identity = BufferIdentity::mint();
        let slot = arena.declare_buffer(BufferDecl {
            id: identity,
            width: 4,
            height: 3,
        });
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        // Two gathers into the same declared slot — the shape a glyph's
        // winding sum has, reading its piece table more than once — so the
        // dedup-by-original-slot in `decode` is actually exercised and not
        // vacuously true for a single reference.
        let a = arena.push_gather(slot, x, y);
        let b = arena.push_gather(slot, y, x);
        let root = arena.push_binary(OpKind::Add, a, b);

        // Real, non-uniform, non-zero contents — the point of this test is
        // that these exact values, not just the 4x3 shape, survive the
        // round trip. A vector of zeros could not tell a dropped payload
        // from a correctly empty one.
        let data: Vec<f32> = (0..12).map(|i| -3.5 + i as f32 * 1.25).collect();

        let kernel = CollapseKernel {
            name: "buffer_gather_test".to_string(),
            family: "buffer".to_string(),
            arena,
            root,
            extent: [64, 4],
            buffer_data: vec![Some(Arc::new(data.clone()))],
        };

        let text = encode(&kernel);
        assert!(
            text.contains("\nB "),
            "encoding a buffer-declaring kernel must carry a `B` line:\n{text}"
        );
        assert!(
            text.contains("\nD "),
            "encoding a buffer-declaring kernel with captured contents must carry a `D` line:\n{text}"
        );
        // Written once per declared slot, not once per `Buffer` leaf — this
        // kernel has two gathers into the one slot.
        assert_eq!(
            text.matches("\nD ").count(),
            1,
            "a buffer's contents must be written once per slot, not once per gather:\n{text}"
        );

        let dir = std::env::temp_dir().join(format!(
            "pixelflow-collapse-corpus-buffer-{}",
            std::process::id()
        ));
        write_dir(&dir, std::slice::from_ref(&kernel));
        let decoded = &read_dir(&dir)[0];
        assert_eq!(
            text,
            encode(decoded),
            "a buffer-declaring kernel did not round trip"
        );
        std::fs::remove_dir_all(&dir).expect("clean up");

        assert_eq!(
            decoded.arena.buffers().len(),
            1,
            "two gathers into one declared slot must decode to one buffer, not two"
        );

        // The buffer's actual contents, not just its shape, survive the
        // round trip exactly — bit for bit, since these values decide the
        // guard masks a `Select` skips or takes at runtime.
        let decoded_data = decoded.buffer_data[0]
            .as_ref()
            .expect("captured buffer contents must survive the round trip, not decode to None");
        assert_eq!(
            **decoded_data, data,
            "buffer contents did not round trip exactly"
        );

        // Bakeable end to end, through the exact path the bench uses:
        // compile at the corpus's own shape, bind the captured buffer of
        // the declared extent, and collapse one real call.
        let mut session = crate::collapse_bench::CollapseSession::open();
        let row = session.measure(decoded, 0);
        assert_eq!(row.kernel, "buffer_gather_test");
        assert!(
            row.measured.ns_median > 0.0,
            "a real collapse call must take non-zero time"
        );
    }
}
