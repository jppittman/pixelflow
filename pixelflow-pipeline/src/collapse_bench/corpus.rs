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

use std::path::Path;

use pixelflow_ir::{
    Environment, ExprBuilder, ExprData, ExprRef, Node, OpKind, Rooted, Term, Uniform,
};

/// One kernel and the lattice it is baked at.
pub struct CollapseKernel {
    /// Unique within a corpus; the row key in the output.
    pub name: String,
    /// Which family it came from — the grouping the analysis reports by.
    pub family: String,
    /// The graph, and the declarations its leaves index. Kept as a pair
    /// because that pair IS what every consumer needs — a `Uniform(0)` leaf
    /// means nothing without the table it indexes — and [`Self::term`] is how
    /// it is handed on.
    pub expr: Rooted<ExprData>,
    pub env: Environment,
    /// The lattice extent, exactly as `Lattice::bake` would see it.
    pub extent: [u32; 2],
}

impl CollapseKernel {
    /// The kernel as the one value the compiler, the optimizer and the
    /// dumper all take.
    #[must_use]
    pub fn term(&self) -> Term<'_> {
        Term::new(self.expr.entry(), &self.env)
    }
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
const HEADER: &str = "# pixelflow collapse corpus v2";
/// The version this format replaced, recognised only so [`decode`] can say
/// *which* mismatch it hit.
const SUPERSEDED_HEADER: &str = "# pixelflow collapse corpus v1";

/// Write `kernels` into `dir`, one `.collapse` file each.
///
/// The node encoding is the arena dumpers' (`pixelflow-core`'s cell-grid
/// dumper, `pixelflow-graphics`'s glyph dumper): reachable nodes in ascending
/// id order with ids remapped dense, constants as bit patterns. The additions
/// are the `family` and `extent` lines — the shape, which is the point.
///
/// # Panics
/// If the directory cannot be created, a file cannot be written, or a kernel
/// contains a node kind the runtime optimizer bails on (`Param`, `Nary`,
/// `Buffer`), which would make the fixture unlike anything production bakes.
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
/// If the directory cannot be read or any file is not a v1 fixture.
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

    let term = kernel.term();
    let root = term.root();
    let dag = term.dag();
    let mut reachable = dag.side_table(false);
    for n in root.descendants() {
        reachable[n] = true;
    }

    let mut out = String::new();
    writeln!(out, "{HEADER}").expect("fmt");
    writeln!(out, "name {}", kernel.name).expect("fmt");
    writeln!(out, "family {}", kernel.family).expect("fmt");
    let [ex, ey] = kernel.extent;
    writeln!(out, "extent {ex} {ey}").expect("fmt");

    // Dense ordinals in ascending, topological (children-before-parents)
    // order over the reachable subgraph. `Node::descendants()` is a
    // parent-first DFS, so it is not a valid dump order on its own — this is
    // the same two-pass shape `expr::encode_into` and the `.arena` dumpers
    // use, for the same reason.
    let mut dense = dag.side_table(None::<u32>);
    let mut next = 0u32;
    for node in dag.iter() {
        if !reachable[node] {
            continue;
        }
        let d = |c: Node<'_, ExprData>| -> u32 { dense[c].expect("child dumped before parent") };
        match *node {
            ExprData::Var(i) => writeln!(out, "V {i}"),
            // A kernel argument. The format knows about one because, with
            // two coordinate axes, a `Uniform` is the *only* leaf that is
            // both invariant across the lattice and beyond the constant
            // folder's reach — which is precisely what the `invariant`
            // family needs to give LICM's frame prologue something to lift.
            // The Z axis used to serve that role; it was the same thing
            // wearing a coordinate's name.
            ExprData::Uniform(u) => {
                writeln!(out, "A {}", term.env().uniform(u).default.to_bits())
            }
            ExprData::Const(bits) => writeln!(out, "C {bits}"),
            ExprData::Op(k) => {
                let children: Vec<Node<'_, ExprData>> = node.children().collect();
                match children.as_slice() {
                    [a] => writeln!(out, "U {k:?} {}", d(*a)),
                    [a, b] => writeln!(out, "Bi {k:?} {} {}", d(*a), d(*b)),
                    [a, b, c] => writeln!(out, "T {k:?} {} {} {}", d(*a), d(*b), d(*c)),
                    _ => panic!(
                        "{}: corpus kernels must be bakeable, but this one holds {k:?} with \
                         {} children",
                        kernel.name,
                        children.len()
                    ),
                }
            }
            other => panic!(
                "{}: corpus kernels must be bakeable, but this one holds {other:?}",
                kernel.name
            ),
        }
        .expect("fmt");
        dense[node] = Some(next);
        next += 1;
    }
    let root_ord = dense[root].expect("the root is its own descendant");
    writeln!(out, "root {root_ord}").expect("fmt");
    out
}

fn decode(path: &Path) -> CollapseKernel {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut lines = text.lines();
    // Name the version mismatch here rather than letting a v1 body surface
    // as "unparseable line": a stale corpus is regenerated, not debugged.
    match lines.next() {
        Some(HEADER) => {}
        Some(SUPERSEDED_HEADER) => panic!(
            "{}: this is a v1 corpus and the format is now v2 — v2 carries \
             an argument node (`A`), which v1 had no way to spell. Regenerate \
             the corpus; a v1 fixture cannot be replayed into a v2 graph.",
            path.display()
        ),
        other => panic!(
            "{}: not a collapse corpus fixture (expected {HEADER:?}, found {other:?})",
            path.display()
        ),
    }

    let mut name = None;
    let mut family = None;
    let mut extent = None;
    let mut root_ord = None;
    let mut b = ExprBuilder::new();
    // Ordinal to the ref it named, in dump order. The dump's ids are dense
    // over the nodes it wrote, children before parents, so a child's ordinal
    // is always already in here when its parent is read.
    let mut refs: Vec<ExprRef> = Vec::new();

    let op = |s: &str| -> OpKind {
        OpKind::all()
            .find(|k| format!("{k:?}") == s)
            .unwrap_or_else(|| panic!("{}: unknown OpKind {s:?}", path.display()))
    };
    let ordinal = |s: &str| -> usize {
        s.parse()
            .unwrap_or_else(|e| panic!("{}: bad id {s:?}: {e}", path.display()))
    };
    let dim = |s: &str| -> u32 {
        s.parse()
            .unwrap_or_else(|e| panic!("{}: bad extent {s:?}: {e}", path.display()))
    };

    for line in lines {
        let f: Vec<&str> = line.split_whitespace().collect();
        // A child ordinal must already have been read: the dump is
        // children-before-parents, so a forward reference is a corrupt file
        // rather than a graph.
        let child = |refs: &[ExprRef], s: &str| -> ExprRef {
            let i = ordinal(s);
            *refs.get(i).unwrap_or_else(|| {
                panic!(
                    "{}: child ordinal {i} is not an already-read node",
                    path.display()
                )
            })
        };
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
                root_ord = Some(ordinal(r));
                continue;
            }
            ["V", i] => b.push_var(i.parse().expect("var index")),
            ["A", bits] => {
                let default = f32::from_bits(bits.parse().expect("argument default bits"));
                let slot = b.declare_uniform(Uniform::new(default).decl());
                b.push_uniform(slot)
            }
            ["C", bits] => b.push_const(f32::from_bits(bits.parse().expect("const bits"))),
            ["U", k, a] => {
                let a = child(&refs, a);
                b.push_unary(op(k), a)
            }
            ["Bi", k, x, y] => {
                let (x, y) = (child(&refs, x), child(&refs, y));
                b.push_binary(op(k), x, y)
            }
            ["T", k, x, y, z] => {
                let (x, y, z) = (child(&refs, x), child(&refs, y), child(&refs, z));
                b.push_ternary(op(k), x, y, z)
            }
            other => panic!("{}: unparseable line {other:?}", path.display()),
        };
        refs.push(pushed);
    }

    let root_ord = root_ord.unwrap_or_else(|| panic!("{}: no root", path.display()));
    let root = *refs
        .get(root_ord)
        .unwrap_or_else(|| panic!("{}: root ordinal {root_ord} names no node", path.display()));
    let (expr, env) = b.finish(&[root]);

    CollapseKernel {
        name: name.unwrap_or_else(|| panic!("{}: no name", path.display())),
        family: family.unwrap_or_else(|| panic!("{}: no family", path.display())),
        expr,
        env,
        extent: extent.unwrap_or_else(|| panic!("{}: no extent", path.display())),
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
    let mut push = |name: String,
                    family: &str,
                    extent: [u32; 2],
                    build: &dyn Fn(&mut ExprBuilder) -> ExprRef| {
        let mut b = ExprBuilder::new();
        let root = build(&mut b);
        let (expr, env) = b.finish(&[root]);
        out.push(CollapseKernel {
            name,
            family: family.to_string(),
            expr,
            env,
            extent,
        });
    };
    for n in [8usize, 16, 32, 64] {
        push(
            format!("wide{n:03}"),
            "wide",
            PRESSURE_EXTENT,
            &move |a: &mut ExprBuilder| wide(a, n),
        );
    }
    for (w, d) in [(8usize, 24usize), (12, 40), (16, 64)] {
        push(
            format!("anchored{w:02}x{d:02}"),
            "anchored",
            PRESSURE_EXTENT,
            &move |a: &mut ExprBuilder| anchored(a, w, d),
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
                &move |a: &mut ExprBuilder| invariants(a, n),
            );
        }
    }
    out
}

/// A leaf that varies in X, salted so the tree is not one common
/// subexpression the optimizer folds away.
fn x_leaf(a: &mut ExprBuilder, salt: usize) -> ExprRef {
    let x = a.push_var(0);
    let c = a.push_const(0.125 + (salt % 13) as f32 * 0.0625);
    a.push_binary(OpKind::Mul, x, c)
}

/// A balanced Add/Sub tree over `n` X-varying leaves.
fn wide(a: &mut ExprBuilder, n: usize) -> ExprRef {
    assert!(n.is_power_of_two(), "wide takes a power of two, got {n}");
    let mut level: Vec<ExprRef> = (0..n).map(|i| x_leaf(a, i)).collect();
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
fn anchored(a: &mut ExprBuilder, w: usize, d: usize) -> ExprRef {
    let anchors: Vec<ExprRef> = (0..w)
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
fn invariants(a: &mut ExprBuilder, n: usize) -> ExprRef {
    let y = a.push_var(1);
    // Frame scope needs a leaf the folder cannot collapse and the lattice
    // cannot vary. That is a kernel argument; it used to be the Z axis,
    // which was the same thing wearing a coordinate's name. A `Const` would
    // fold and leave LICM nothing to lift.
    let arg = a.declare_uniform(Uniform::new(CORPUS_ARG).decl());
    let z = a.push_uniform(arg);
    let terms: Vec<ExprRef> = (0..n)
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
    #[should_panic(expected = "this is a v1 corpus and the format is now v2")]
    fn a_superseded_corpus_names_the_version_it_is() {
        let dir = std::env::temp_dir().join(format!(
            "pixelflow-collapse-corpus-v1-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("stale.txt");
        std::fs::write(
            &path,
            format!("{SUPERSEDED_HEADER}\nname stale\nfamily wide\nextent 8 8\nV 0\nroot 0\n"),
        )
        .expect("write");
        let _ = decode(&path);
    }
}
