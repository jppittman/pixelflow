//! Binary corpus format for pre-parsed expression storage.
//!
//! Replaces JSONL text corpus with a binary format that loads in microseconds
//! via sequential read (no parsing, no allocation beyond the arena vecs).
//!
//! ## Format (the header's second field is a derived identity, not a
//! ## hand-bumped version — see "The schema identity is coupled to the
//! ## payload's meaning" below)
//!
//! ```text
//! magic: [u8; 4] = b"PXCR"
//! schema_identity: u64 (little-endian) = corpus_identity()
//! count: u32 (little-endian)
//!
//! For each expression:
//!   name_len: u16 (little-endian)
//!   name: [u8; name_len]       (UTF-8)
//!   node_count: u32 (le)
//!   nary_count: u32 (le)
//!   root_index: u32 (le)       (ExprId.0)
//!   nodes: node_count encoded ExprNodes (variable per-node)
//!   nary_children: [u32; nary_count] (le) (ExprId.0 values)
//! ```
//!
//! Each ExprNode is encoded as:
//!   tag: u8  (0=Var, 1=Const, 2=Param, 3=Unary, 4=Binary, 5=Ternary, 6=Nary)
//!   payload varies by tag.
//!
//! ## Only the reachable subtree is stored
//!
//! [`write_corpus`] serializes [`reachable_subtree`], not the caller's whole
//! arena. Generator arenas are append-only scratch space: rewriting a node
//! pushes a replacement and abandons the original, so an arena holding a
//! 12-node expression can carry hundreds of dead nodes. Storing them made
//! `arena.len()` a measure of the *generator's history* rather than of the
//! expression, and every downstream size filter that read `arena.len()`
//! silently dropped small expressions with long provenance — non-randomly,
//! since dead-node count correlates with how many rewrite passes ran (the
//! B3 holdout-integrity bug: 110 of 380 DEV entries dropped by a filter
//! measuring dead nodes). After compaction, stored size *is* expression size.
//!
//! ## The schema identity is coupled to the payload's meaning
//!
//! Three independent things can invalidate a stored corpus, and none of them
//! announce themselves as a parse error: the op encoding (`pixelflow-ir`'s
//! `OpKind::marshal` is free to renumber without telling anyone, so a stale
//! corpus decodes to different operations rather than failing to parse); the
//! reachable-subtree compaction (an uncompacted file parses fine, but its
//! node counts are arena history, not expression size — bug B3); and which
//! tier a file was deduplicated against (a corpus keyed on raw structure
//! instead of the feature-quotient [`FenceKey`](super::structural::FenceKey)
//! parses fine but can hold a DEV/FINAL leak in TRAIN — P1(d)). A hand-bumped
//! `VERSION` integer needs a human to remember to move it every time any of
//! the three changes; this format instead carries [`corpus_identity`], which
//! folds [`crate::schema::SchemaIdentity`]'s content hash of
//! `CorpusFormat::SCHEMA` (editing that description IS the version bump, so
//! there is no separate step to forget) together with a hash of the LIVE
//! `OpKind` encoding table — the op-renumbering case that prose alone cannot
//! see, since "dense 0..COUNT discriminants" reads the same before and after
//! a renumbering (docs/plans/2026-08-17-cost-model-domain.md, J9, kills
//! P1(b)). Any stored identity other than the current one is a hard load
//! error.

use std::io::{self, Write};
use std::path::Path;

use pixelflow_ir::fold::Fold;
use pixelflow_ir::kind::OpCode;
use pixelflow_ir::{ExprArena, ExprId, ExprNode, OpKind};

use crate::schema::{SchemaIdentity, fnv1a64_const, identity_mismatch};

/// Marker type naming the corpus binary format for [`SchemaIdentity`]. The
/// format has no single Rust value of its own — it serializes a
/// `Vec<(String, ExprArena, ExprId)>` — so this type exists purely to carry
/// `SCHEMA` and derive `SCHEMA_IDENTITY` from it.
struct CorpusFormat;

impl SchemaIdentity for CorpusFormat {
    const MAGIC: &'static str = "PXCR";
    // Every field this format's bytes carry, and what each one means. This
    // text IS the version: change what a field means here and
    // `SCHEMA_IDENTITY` (its content hash) changes with it, so a corpus
    // written under the old meaning cannot silently be read under the new
    // one. History, for readers orienting themselves: the op encoding went
    // dense (0..COUNT) under `OpKind::marshal`; entries were narrowed to only
    // the subtree reachable from `root` (arena.len() is generator history,
    // not expression size — bug B3); and the cross-tier dedup key moved from
    // raw structure to the feature-quotient `FenceKey`, because the
    // extraction head's features collapse literals and a literal-keyed dedup
    // could seat a DEV/FINAL-equivalent expression in TRAIN (P1(d)).
    const SCHEMA: &'static str = "\
        header: magic[4]=PXCR, schema_identity: u64 le, count: u32 le entries follow; \
        entry: name_len u16 le, name utf8 bytes, node_count u32 le, \
        root_index u32 le (ExprId.0 into this entry's own node list), \
        nodes: node_count encoded ExprNodes in child-before-parent order; \
        ExprNode tag byte: 0=Var(index u8), 1=Const(f32 le), 2=Param(index u8), \
        3=Unary(OpKind::marshal, ExprId le), \
        4=Binary(OpKind::marshal, ExprId le, ExprId le), \
        5=Ternary(OpKind::marshal, ExprId le, ExprId le, ExprId le), \
        6=Nary(OpKind::marshal, len u16 le, [ExprId le; len]), \
        7=Buffer(BufferId u16 le) refused at write time, its declaration is never \
        serialized; \
        op byte encoding: pixelflow_ir::OpKind::marshal, dense 0..COUNT discriminants \
        private to pixelflow-ir; \
        entries store ONLY the subtree reachable from root (reachable_subtree \
        compaction) so stored node_count is expression size, not generator-arena \
        size; \
        cross-tier dedup key an on-disk TRAIN tier was built against: the \
        feature-quotient FenceKey (structural_key composed with the extraction \
        head's literal-collapsing feature map), not raw structural equality";
}

fn regen_command() -> &'static str {
    "cargo run --release -p pixelflow-pipeline --features training --bin gen_bench_corpus"
}

// ── Identity: SCHEMA text folded with the LIVE op encoding table ───────────
//
// `CorpusFormat::SCHEMA_IDENTITY` alone has a hole: it hashes the prose in
// `SCHEMA` above, and that prose only *describes* the encoding as "dense
// 0..COUNT discriminants private to pixelflow-ir" — a sentence that stays
// true, and therefore stays byte-identical, no matter which op ends up at
// which byte. `OpKind::marshal` is free to renumber (`docs/designs/
// opkind-numbering-is-private.md`), and a renumbering changes nothing this
// text says, so `SCHEMA_IDENTITY` alone cannot see it: an old corpus would
// pass the new identity check and silently decode every affected opcode as
// the wrong operation — exactly the failure J9 exists to make impossible.
//
// This is the derived-encoding-fingerprint design that opkind-numbering-is-
// private.md §4.3 named `OpKind::ENCODING_ID` and rejected as disproportion-
// ate — at the time, a hand-bumped `VERSION` integer was still the actual
// gate, so the fingerprint would have been a second, redundant guard over
// the same format. That premise is gone: this corpus format's gate is now
// `SchemaIdentity`'s derived hash and nothing else, so a renumbering that
// outruns the prose has no other guard left to catch it. `pixelflow-ir` is
// out of scope for this change (its numbering stays private, per that same
// doc), so the fingerprint is computed here, from `OpKind`'s public
// `all()`/`marshal()`/`name()` surface, rather than as an `OpKind` const.
//
// Folds `(name, marshal() byte)` for every live op, in `OpKind::all()`
// order, plus the op count — not just the byte sequence, which is always
// `0..COUNT` by construction and so is invariant under a renumbering that
// only swaps which op sits at which position.
fn opcode_encoding_identity() -> u64 {
    let mut buf: Vec<u8> = Vec::new();
    let mut count: u32 = 0;
    for op in OpKind::all() {
        let name = op.name().as_bytes();
        assert!(
            name.len() <= u8::MAX as usize,
            "opcode_encoding_identity: op name '{}' exceeds u8::MAX bytes",
            op.name()
        );
        buf.push(name.len() as u8);
        buf.extend_from_slice(name);
        buf.extend_from_slice(&op.marshal().to_bytes());
        count += 1;
    }
    buf.extend_from_slice(&count.to_le_bytes());
    fnv1a64_const(&buf)
}

/// The identity actually written to disk and checked on load: the schema
/// prose's hash, folded with the live `(op name, encoded byte)` table so a
/// renumbering in `pixelflow-ir` changes the identity by construction, not
/// by someone remembering to edit `CorpusFormat::SCHEMA` to match.
fn corpus_identity() -> u64 {
    let mut buf = Vec::with_capacity(16);
    buf.extend_from_slice(&CorpusFormat::SCHEMA_IDENTITY.to_le_bytes());
    buf.extend_from_slice(&opcode_encoding_identity().to_le_bytes());
    fnv1a64_const(&buf)
}

// ── ExprNode serialization tags ──────────────────────────────────────────────

const TAG_VAR: u8 = 0;
const TAG_CONST: u8 = 1;
const TAG_PARAM: u8 = 2;
const TAG_UNARY: u8 = 3;
const TAG_BINARY: u8 = 4;
const TAG_TERNARY: u8 = 5;
const TAG_NARY: u8 = 6;
const TAG_REDUCE: u8 = 9;
const TAG_BUFFER: u8 = 7;

// ── Reachable-subtree compaction ─────────────────────────────────────────────

/// Rebuild `(arena, root)` keeping only the nodes reachable from `root`.
///
/// DAG sharing is preserved: a node referenced from several parents is
/// emitted once and referenced by the same new id, so compaction never
/// expands a shared subgraph into a tree. Children are emitted before their
/// parents, so the result satisfies [`ExprArena::from_raw`]'s ordering
/// contract.
///
/// This is what makes stored size mean expression size. Callers that
/// generate expressions into a long-lived scratch arena (`BwdGenerator`, the
/// junkify passes, e-graph extraction) leave dead nodes behind by design —
/// an append-only arena has no other way to rewrite — and `arena.len()`
/// counts those. Any consumer that treats `arena.len()` as "how big is this
/// expression" is measuring the wrong population.
///
/// The buffer table is NOT carried over: the corpus format has never
/// serialized buffer declarations, and this function exists to serve it.
/// Compacting an arena with `Buffer` leaves would therefore produce an entry
/// whose declarations were already going to be lost, so it is refused here
/// rather than written out silently broken.
///
/// # Panics
///
/// Panics if the reachable subgraph contains a `Buffer` node.
#[must_use]
pub fn reachable_subtree(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    enum Task {
        Descend(ExprId),
        Emit(ExprId),
    }

    let mut id_map: Vec<Option<ExprId>> = vec![None; arena.len()];
    let mut out_arena = ExprArena::new();
    let mut work: Vec<Task> = vec![Task::Descend(root)];

    while let Some(task) = work.pop() {
        match task {
            Task::Descend(id) => {
                if id_map[id.0 as usize].is_some() {
                    continue;
                }
                work.push(Task::Emit(id));
                let children: Vec<ExprId> = arena.children(id).collect();
                for child in children.into_iter().rev() {
                    work.push(Task::Descend(child));
                }
            }
            Task::Emit(id) => {
                if id_map[id.0 as usize].is_some() {
                    continue;
                }
                let map = |old: ExprId| -> ExprId {
                    id_map[old.0 as usize]
                        .expect("reachable_subtree: child must be emitted before its parent")
                };
                let new_id = match arena.node(id) {
                    ExprNode::Var(i) => out_arena.push_var(*i),
                    ExprNode::Const(v) => out_arena.push_const(*v),
                    ExprNode::Param(i) => out_arena.push_param(*i),
                    ExprNode::Buffer(b) => panic!(
                        "reachable_subtree: expression references Buffer({}), whose declaration \
                         the corpus format does not serialize — writing it would store a node \
                         that cannot be read back correctly",
                        b.0
                    ),
                    ExprNode::Uniform(u) => panic!(
                        "reachable_subtree: expression references Uniform({}), whose declaration \
                         the corpus format does not serialize",
                        u.0
                    ),
                    ExprNode::Ref(k) => panic!(
                        "reachable_subtree: expression references Ref({k:?}), which names a \
                         kernel interned in this process — a corpus outlives the process, so \
                         the key would read back naming nothing"
                    ),
                    ExprNode::Unary(op, a) => out_arena.push_unary(*op, map(*a)),
                    ExprNode::Binary(op, a, b) => out_arena.push_binary(*op, map(*a), map(*b)),
                    ExprNode::Ternary(op, a, b, c) => {
                        out_arena.push_ternary(*op, map(*a), map(*b), map(*c))
                    }
                    ExprNode::Reduce { fold, body } => out_arena.push_reduce(*fold, map(*body)),
                    ExprNode::Nary(op, _, _) => {
                        let mapped_children: Vec<ExprId> = arena.children(id).map(map).collect();
                        out_arena.push_nary(*op, &mapped_children)
                    }
                };
                id_map[id.0 as usize] = Some(new_id);
            }
        }
    }

    let new_root = id_map[root.0 as usize].expect("reachable_subtree: root was never emitted");
    (out_arena, new_root)
}

// ── Write ────────────────────────────────────────────────────────────────────

/// Write a binary corpus to `path`.
///
/// Each entry is compacted with [`reachable_subtree`] before serialization,
/// so the stored node count is the expression's size, not its arena's.
///
/// # Panics
///
/// Panics if any expression name exceeds `u16::MAX` bytes, or if any
/// expression references a `Buffer` node (see [`reachable_subtree`]).
pub fn write_corpus(path: &Path, entries: &[(String, ExprArena, ExprId)]) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut w = io::BufWriter::new(file);

    // Header
    w.write_all(CorpusFormat::MAGIC.as_bytes())?;
    w.write_all(&corpus_identity().to_le_bytes())?;
    w.write_all(&(entries.len() as u32).to_le_bytes())?;

    for (name, arena, root) in entries {
        write_entry(&mut w, name, arena, *root)?;
    }

    w.flush()?;
    Ok(())
}

fn write_entry(w: &mut impl Write, name: &str, arena: &ExprArena, root: ExprId) -> io::Result<()> {
    let name_bytes = name.as_bytes();
    assert!(
        name_bytes.len() <= u16::MAX as usize,
        "write_corpus: expression name exceeds u16::MAX bytes: '{}'",
        name
    );

    // Compact first: the caller's arena is generator scratch space and its
    // dead nodes are not part of this expression (format v3).
    let (compact, compact_root) = reachable_subtree(arena, root);
    let count = compact.len();

    w.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
    w.write_all(name_bytes)?;
    w.write_all(&(count as u32).to_le_bytes())?;
    w.write_all(&compact_root.0.to_le_bytes())?;

    // Nodes
    for idx in 0..count {
        let id = ExprId(idx as u32);
        write_node(w, &compact, id)?;
    }

    Ok(())
}

fn write_node(w: &mut impl Write, arena: &ExprArena, id: ExprId) -> io::Result<()> {
    match arena.node(id) {
        ExprNode::Var(i) => {
            w.write_all(&[TAG_VAR, *i])?;
        }
        ExprNode::Const(v) => {
            w.write_all(&[TAG_CONST])?;
            w.write_all(&v.to_le_bytes())?;
        }
        ExprNode::Param(i) => {
            w.write_all(&[TAG_PARAM, *i])?;
        }
        ExprNode::Unary(op, a) => {
            w.write_all(&[TAG_UNARY])?;
            w.write_all(&op.marshal().to_bytes())?;
            w.write_all(&a.0.to_le_bytes())?;
        }
        ExprNode::Binary(op, a, b) => {
            w.write_all(&[TAG_BINARY])?;
            w.write_all(&op.marshal().to_bytes())?;
            w.write_all(&a.0.to_le_bytes())?;
            w.write_all(&b.0.to_le_bytes())?;
        }
        ExprNode::Ternary(op, a, b, c) => {
            w.write_all(&[TAG_TERNARY])?;
            w.write_all(&op.marshal().to_bytes())?;
            w.write_all(&a.0.to_le_bytes())?;
            w.write_all(&b.0.to_le_bytes())?;
            w.write_all(&c.0.to_le_bytes())?;
        }
        ExprNode::Nary(op, _, _) => {
            w.write_all(&[TAG_NARY])?;
            w.write_all(&op.marshal().to_bytes())?;
            let children = arena.children(id);
            let len = children.len();
            assert!(
                len <= u16::MAX as usize,
                "write_node: Nary children count exceeds u16::MAX"
            );
            w.write_all(&(len as u16).to_le_bytes())?;
            for child in children {
                w.write_all(&child.0.to_le_bytes())?;
            }
        }
        ExprNode::Buffer(b) => {
            w.write_all(&[TAG_BUFFER])?;
            w.write_all(&b.0.to_le_bytes())?;
        }
        // The fold as opaque bits: metadata, not children.
        ExprNode::Reduce { fold, body } => {
            w.write_all(&[TAG_REDUCE])?;
            w.write_all(&fold.to_bits().to_le_bytes())?;
            w.write_all(&body.0.to_le_bytes())?;
        }
        ExprNode::Uniform(u) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Uniform({}) has no corpus encoding: its declaration is not serialized",
                    u.0
                ),
            ));
        }
        // A key names an entry in *this* process's `KernelStore`, and a
        // corpus outlives the process — encoding one would store a name
        // nothing can resolve on the way back in. Expand refs before writing.
        ExprNode::Ref(k) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Ref({k:?}) has no corpus encoding: it names a process-local kernel"),
            ));
        }
    }
    Ok(())
}

// ── Read ─────────────────────────────────────────────────────────────────────

/// Read a binary corpus from `path`.
///
/// Returns `(name, arena, root)` triples.
pub fn read_corpus(path: &Path) -> io::Result<Vec<(String, ExprArena, ExprId)>> {
    let data = std::fs::read(path)?;
    read_corpus_bytes(&data)
}

fn read_corpus_bytes(data: &[u8]) -> io::Result<Vec<(String, ExprArena, ExprId)>> {
    let mut r = Cursor::new(data);

    // Header
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if magic != *CorpusFormat::MAGIC.as_bytes() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "bad corpus magic: expected {:?}, got {:?}",
                CorpusFormat::MAGIC.as_bytes(),
                magic
            ),
        ));
    }

    // Exact-identity check, no tolerance. Op bytes mean whatever the IR's
    // encoding said when the file was written, and that encoding may change
    // without notice — so a "best effort" read of a stale file would decode
    // valid-looking bytes into the wrong ops rather than fail. A hand-bumped
    // integer needs a human to remember to move it every time the format's
    // meaning changes; this identity is derived from `CorpusFormat::SCHEMA`
    // folded with the live op encoding table (`corpus_identity`, docs/plans/
    // 2026-08-17-cost-model-domain.md, J9), so any change to what the bytes
    // below mean — INCLUDING a silent `OpKind::marshal` renumbering the
    // prose doesn't happen to mention — changes the identity by
    // construction, not by someone remembering to update `SCHEMA`'s text.
    let stored_identity = r.read_u64()?;
    let expected_identity = corpus_identity();
    if stored_identity != expected_identity {
        return Err(identity_mismatch(
            "corpus",
            stored_identity,
            expected_identity,
            regen_command(),
        ));
    }

    let count = r.read_u32()? as usize;
    let mut entries = Vec::with_capacity(count);

    for i in 0..count {
        let entry = read_entry(&mut r)
            .map_err(|e| io::Error::new(e.kind(), format!("corpus entry {i}/{count}: {e}")))?;
        entries.push(entry);
    }

    Ok(entries)
}

fn read_entry(r: &mut Cursor<'_>) -> io::Result<(String, ExprArena, ExprId)> {
    let name_len = r.read_u16()? as usize;
    let name = {
        let bytes = r.read_bytes(name_len)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid UTF-8 name: {e}"),
            )
        })?
    };

    let node_count = r.read_u32()? as usize;
    let root_index = r.read_u32()?;

    let mut arena = ExprArena::new();
    for _ in 0..node_count {
        read_node_into(r, &mut arena)?;
    }

    let root = ExprId(root_index);

    Ok((name, arena, root))
}

fn read_node_into(r: &mut Cursor<'_>, arena: &mut ExprArena) -> io::Result<ExprId> {
    let tag = r.read_u8()?;
    match tag {
        TAG_VAR => {
            let i = r.read_u8()?;
            Ok(arena.push_var(i))
        }
        TAG_CONST => {
            let bits = r.read_u32()?;
            Ok(arena.push_const(f32::from_le_bytes(bits.to_le_bytes())))
        }
        TAG_PARAM => {
            let i = r.read_u8()?;
            Ok(arena.push_param(i))
        }
        TAG_UNARY => {
            let op = read_opkind(r)?;
            let a = ExprId(r.read_u32()?);
            Ok(arena.push_unary(op, a))
        }
        TAG_BINARY => {
            let op = read_opkind(r)?;
            let a = ExprId(r.read_u32()?);
            let b = ExprId(r.read_u32()?);
            Ok(arena.push_binary(op, a, b))
        }
        TAG_TERNARY => {
            let op = read_opkind(r)?;
            let a = ExprId(r.read_u32()?);
            let b = ExprId(r.read_u32()?);
            let c = ExprId(r.read_u32()?);
            Ok(arena.push_ternary(op, a, b, c))
        }
        TAG_NARY => {
            let op = read_opkind(r)?;
            let len = r.read_u16()? as usize;
            let mut children = Vec::with_capacity(len);
            for _ in 0..len {
                children.push(ExprId(r.read_u32()?));
            }
            Ok(arena.push_nary(op, &children))
        }
        TAG_BUFFER => {
            let b = pixelflow_ir::arena::BufferId(r.read_u16()?);
            Ok(arena.push_buffer(b))
        }
        TAG_REDUCE => {
            let bits = r.read_u64()?;
            let fold = Fold::from_bits(bits).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("corpus fold bits {bits} name no fold"),
                )
            })?;
            let body = ExprId(r.read_u32()?);
            Ok(arena.push_reduce(fold, body))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown ExprNode tag: {tag}"),
        )),
    }
}

fn read_opkind(r: &mut Cursor<'_>) -> io::Result<OpKind> {
    let mut bytes = [0u8; OpCode::SIZE];
    for b in &mut bytes {
        *b = r.read_u8()?;
    }
    OpKind::unmarshal(OpCode::from_bytes(bytes)).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("no op is encoded by {bytes:?}"),
        )
    })
}

// ── Minimal cursor for zero-copy reads ───────────────────────────────────────

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        let end = self.pos + buf.len();
        if end > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "read_exact: need {} bytes at offset {}, but only {} remain",
                    buf.len(),
                    self.pos,
                    self.data.len() - self.pos
                ),
            ));
        }
        buf.copy_from_slice(&self.data[self.pos..end]);
        self.pos = end;
        Ok(())
    }

    fn read_bytes(&mut self, n: usize) -> io::Result<&'a [u8]> {
        let end = self.pos + n;
        if end > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "read_bytes: need {n} bytes at offset {}, but only {} remain",
                    self.pos,
                    self.data.len() - self.pos
                ),
            ));
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> io::Result<u8> {
        if self.pos >= self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("read_u8: at offset {}, no bytes remain", self.pos),
            ));
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_u16(&mut self) -> io::Result<u16> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_u32(&mut self) -> io::Result<u32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> io::Result<u64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Per-process path: concurrent `cargo test` runs must not share corpus files,
    // or one process's remove_file races another's write/read.
    fn unique_tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("corpus_rt_{name}_{}.bin", std::process::id()))
    }

    #[test]
    fn round_trip_empty() {
        let tmp = unique_tmp("empty");
        let entries: Vec<(String, ExprArena, ExprId)> = Vec::new();
        write_corpus(&tmp, &entries).expect("write");
        let loaded = read_corpus(&tmp).expect("read");
        assert!(loaded.is_empty());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn round_trip_simple() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let root = arena.push_binary(OpKind::Add, x, y);

        let entries = vec![("test_add".to_string(), arena, root)];

        let tmp = unique_tmp("simple");
        write_corpus(&tmp, &entries).expect("write");
        let loaded = read_corpus(&tmp).expect("read");

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "test_add");
        assert_eq!(loaded[0].1.len(), 3);
        assert_eq!(loaded[0].2.0, root.0);

        // Verify node equality
        for (i, node) in entries[0].1.nodes_raw().iter().enumerate() {
            assert_eq!(
                node,
                loaded[0].1.node(ExprId(i as u32)),
                "node {i} mismatch"
            );
        }

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn round_trip_with_const_and_unary() {
        let mut arena = ExprArena::new();
        let c = arena.push_const(std::f32::consts::PI);
        let root = arena.push_unary(OpKind::Sqrt, c);

        let entries = vec![("sqrt_pi".to_string(), arena, root)];

        let tmp = unique_tmp("unary");
        write_corpus(&tmp, &entries).expect("write");
        let loaded = read_corpus(&tmp).expect("read");

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "sqrt_pi");
        // Check the const value round-trips
        match loaded[0].1.node(ExprId(0)) {
            ExprNode::Const(v) => assert!(
                (v - std::f32::consts::PI).abs() < 1e-6,
                "const mismatch: {v}"
            ),
            other => panic!("expected Const, got {other:?}"),
        }

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn round_trip_ternary() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let z = arena.push_var(2);
        let root = arena.push_ternary(OpKind::Select, x, y, z);

        let entries = vec![("select_xyz".to_string(), arena, root)];

        let tmp = unique_tmp("ternary");
        write_corpus(&tmp, &entries).expect("write");
        let loaded = read_corpus(&tmp).expect("read");

        assert_eq!(loaded.len(), 1);
        match loaded[0].1.node(loaded[0].2) {
            ExprNode::Ternary(OpKind::Select, _, _, _) => {}
            other => panic!("expected Ternary(Select,...), got {other:?}"),
        }

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn round_trip_nary() {
        let mut arena = ExprArena::new();
        let a = arena.push_var(0);
        let b = arena.push_var(1);
        let c = arena.push_var(2);
        let root = arena.push_nary(OpKind::Tuple, &[a, b, c]);

        let entries = vec![("tuple_abc".to_string(), arena, root)];

        let tmp = unique_tmp("nary");
        write_corpus(&tmp, &entries).expect("write");
        let loaded = read_corpus(&tmp).expect("read");

        assert_eq!(loaded.len(), 1);
        match loaded[0].1.node(loaded[0].2) {
            ExprNode::Nary(OpKind::Tuple, _, _) => {
                let children: Vec<_> = loaded[0].1.children(loaded[0].2).collect();
                assert_eq!(children.len(), 3);
            }
            other => panic!("expected Nary(Tuple,...), got {other:?}"),
        }

        let _ = std::fs::remove_file(&tmp);
    }

    // Header-rejection tests drive the reader through its public entry point:
    // `read_corpus_bytes` is private, and pinning it here would test a path no
    // caller can reach. `name` keeps sibling tests off each other's fixture
    // within a process, `unique_tmp` keeps concurrent test processes apart.
    fn read_corpus_from_bytes(
        name: &str,
        data: &[u8],
    ) -> io::Result<Vec<(String, ExprArena, ExprId)>> {
        let tmp = unique_tmp(name);
        std::fs::write(&tmp, data).expect("write fixture");
        let result = read_corpus(&tmp);
        let _ = std::fs::remove_file(&tmp);
        result
    }

    #[test]
    fn bad_magic_fails() {
        let data = b"BADMxxxxxxxx";
        match read_corpus_from_bytes("bad_magic", data) {
            Ok(_) => panic!("expected error for bad magic"),
            Err(e) => assert!(
                e.to_string().contains("bad corpus magic"),
                "unexpected error: {e}"
            ),
        }
    }

    #[test]
    fn stale_schema_identity_is_refused_with_regeneration_hint() {
        // Any identity other than the current one must be a hard error — this
        // is what replaced three separate hand-bumped-version tests (op
        // renumbering, subtree compaction, FenceKey dedup): all three are now
        // exactly one case, "the stored identity doesn't match", because the
        // identity is derived from the format description rather than
        // hand-maintained beside it (J9).
        let mut data = Vec::new();
        data.extend_from_slice(CorpusFormat::MAGIC.as_bytes());
        data.extend_from_slice(&0xdead_beef_dead_beefu64.to_le_bytes()); // stale identity
        data.extend_from_slice(&0u32.to_le_bytes()); // count=0
        match read_corpus_from_bytes("stale_identity", &data) {
            Ok(_) => panic!("a stale schema identity must be refused, not decoded"),
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("schema identity mismatch"),
                    "error must name the failure class: {msg}"
                );
                assert!(
                    msg.contains("deadbeefdeadbeef"),
                    "error must name the rejected identity: {msg}"
                );
                assert!(
                    msg.contains("gen_bench_corpus"),
                    "error must name the regeneration binary: {msg}"
                );
            }
        }
    }

    #[test]
    fn schema_identity_changes_when_the_schema_text_does() {
        // The mechanism this format now relies on, pinned directly: editing
        // what a field means (here, simulated by comparing two schema
        // strings) changes the derived identity without a separate bump step
        // to forget.
        let old_meaning = fnv1a64_const(b"root_index means an index into the WRITER's arena");
        let new_meaning = fnv1a64_const(b"root_index means an index into THIS ENTRY's nodes");
        assert_ne!(old_meaning, new_meaning);
        assert_eq!(CorpusFormat::SCHEMA_IDENTITY, CorpusFormat::SCHEMA_IDENTITY);
    }

    // ── corpus_identity folds in the live opcode encoding (P1 finding on
    // PR #1019: SCHEMA_IDENTITY alone hashes only prose) ───────────────────

    #[test]
    fn corpus_identity_is_deterministic() {
        assert_eq!(corpus_identity(), corpus_identity());
        assert_eq!(opcode_encoding_identity(), opcode_encoding_identity());
    }

    #[test]
    fn corpus_identity_depends_on_more_than_the_schema_text() {
        // The defect this guards: `CorpusFormat::SCHEMA` describes the op
        // encoding only as "dense 0..COUNT discriminants" — a sentence a
        // renumbering never has to touch. If `corpus_identity` reduced to
        // `SCHEMA_IDENTITY` alone, that renumbering would leave the on-disk
        // gate unchanged and a stale corpus would decode every affected op
        // as whatever now sits at its old byte. Folding the live encoding
        // table in means the composite identity cannot equal the bare
        // schema-text hash.
        assert_ne!(
            corpus_identity(),
            CorpusFormat::SCHEMA_IDENTITY,
            "corpus_identity must depend on the live opcode encoding, not just the \
             schema prose describing it"
        );
    }

    #[test]
    fn encoding_identity_changes_if_an_op_is_reassigned_a_different_byte() {
        // Pins the sensitivity a renumbering needs caught: hashing the byte
        // sequence ALONE would be invariant under a swap (marshal's bytes
        // are always the dense sequence 0..COUNT no matter which op holds
        // which position), so the identity has to be computed over
        // (name, byte) pairs — this reimplements that computation generically
        // over a synthetic table, standing in for a renumbering of the real
        // (private) `OpKind` table, which nothing outside `pixelflow-ir` can
        // fabricate directly.
        fn identity_of(table: &[(&str, u8)]) -> u64 {
            let mut buf: Vec<u8> = Vec::new();
            for (name, byte) in table {
                buf.push(name.len() as u8);
                buf.extend_from_slice(name.as_bytes());
                buf.push(*byte);
            }
            buf.extend_from_slice(&(table.len() as u32).to_le_bytes());
            fnv1a64_const(&buf)
        }

        let before = [("add", 0u8), ("sub", 1u8), ("mul", 2u8)];
        // A renumbering: `add` and `sub` swap encoded bytes. The byte
        // sequence itself (0,1,2) is unchanged; only the mapping is.
        let after = [("add", 1u8), ("sub", 0u8), ("mul", 2u8)];

        assert_ne!(
            identity_of(&before),
            identity_of(&after),
            "swapping which op maps to a byte must change the identity — this is \
             exactly what OpKind::marshal renumbering does, and a hash that can't \
             see it defeats the corpus identity check (P1 on PR #1019)"
        );
    }

    // ── Reachable-subtree compaction (bug B3) ───────────────────────────────

    /// An arena carrying the dead nodes an append-only generator leaves
    /// behind: `X + 1.0` is built, then abandoned in favour of `X * 2.0`.
    fn arena_with_dead_nodes() -> (ExprArena, ExprId) {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let one = arena.push_const(1.0);
        let _abandoned = arena.push_binary(OpKind::Add, x, one);
        let two = arena.push_const(2.0);
        let root = arena.push_binary(OpKind::Mul, x, two);
        (arena, root)
    }

    #[test]
    fn compaction_drops_dead_nodes_and_preserves_the_expression() {
        let (arena, root) = arena_with_dead_nodes();
        assert_eq!(arena.len(), 5, "fixture should carry 2 dead nodes");
        assert_eq!(arena.node_count_subtree(root), 3);

        let (compact, compact_root) = reachable_subtree(&arena, root);
        assert_eq!(compact.len(), 3, "only the reachable DAG survives");
        assert_eq!(compact.node_count_subtree(compact_root), 3);
        assert!(
            compact.subtree_eq(compact_root, &arena, root),
            "compaction must preserve the expression, not just its size"
        );
    }

    #[test]
    fn compaction_preserves_dag_sharing() {
        // `s + s` where `s = sqrt(X)`: the shared child must stay shared, or
        // compaction would turn a DAG into an exponentially larger tree.
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let s = arena.push_unary(OpKind::Sqrt, x);
        let root = arena.push_binary(OpKind::Add, s, s);

        let (compact, compact_root) = reachable_subtree(&arena, root);
        assert_eq!(compact.len(), 3, "shared node must be emitted once");
        match compact.node(compact_root) {
            ExprNode::Binary(OpKind::Add, a, b) => {
                assert_eq!(a, b, "both operands must reference the same node");
            }
            other => panic!("expected Binary(Add, ..), got {other:?}"),
        }
    }

    #[test]
    fn stored_size_reflects_the_expression_not_the_arena() {
        // The B3 regression: a small expression in a junk-heavy arena must
        // round-trip as a small expression. Reading back `arena.len() == 5`
        // is what let a `> N` filter drop 29% of the DEV tier.
        let (arena, root) = arena_with_dead_nodes();
        let entries = vec![("junky".to_string(), arena.clone(), root)];

        let tmp = unique_tmp("dead_nodes");
        write_corpus(&tmp, &entries).expect("write");
        let loaded = read_corpus(&tmp).expect("read");
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(loaded.len(), 1);
        let (_, loaded_arena, loaded_root) = &loaded[0];
        assert_eq!(
            loaded_arena.len(),
            3,
            "stored arena must hold only the reachable subtree, got {} nodes",
            loaded_arena.len()
        );
        assert_eq!(loaded_arena.node_count_subtree(*loaded_root), 3);
        assert!(
            loaded_arena.subtree_eq(*loaded_root, &arena, root),
            "the round-tripped expression must equal the original"
        );
    }

    #[test]
    #[should_panic(expected = "does not serialize")]
    fn compaction_refuses_buffer_nodes() {
        // A Buffer leaf's declaration is not part of the corpus format, so a
        // corpus entry holding one is unreadable-by-construction. Refuse at
        // write time rather than storing a node that decodes to nothing.
        let mut arena = ExprArena::new();
        let buf = arena.declare_buffer(pixelflow_ir::arena::BufferDecl {
            id: pixelflow_ir::arena::BufferIdentity::mint(),
            width: 8,
            height: 8,
        });
        let root = arena.push_buffer(buf);
        let _ = reachable_subtree(&arena, root);
    }

    #[test]
    fn round_trip_multiple_entries() {
        let mut entries = Vec::new();

        // Entry 1: X + Y
        let mut a1 = ExprArena::new();
        let x = a1.push_var(0);
        let y = a1.push_var(1);
        let r1 = a1.push_binary(OpKind::Add, x, y);
        entries.push(("add_xy".to_string(), a1, r1));

        // Entry 2: sqrt(pi)
        let mut a2 = ExprArena::new();
        let c = a2.push_const(std::f32::consts::PI);
        let r2 = a2.push_unary(OpKind::Sqrt, c);
        entries.push(("sqrt_pi".to_string(), a2, r2));

        let tmp = unique_tmp("multi");
        write_corpus(&tmp, &entries).expect("write");
        let loaded = read_corpus(&tmp).expect("read");

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].0, "add_xy");
        assert_eq!(loaded[1].0, "sqrt_pi");
        assert_eq!(loaded[0].1.len(), 3);
        assert_eq!(loaded[1].1.len(), 2);

        let _ = std::fs::remove_file(&tmp);
    }
}
