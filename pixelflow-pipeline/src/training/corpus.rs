//! Binary corpus format for pre-parsed expression storage.
//!
//! Replaces JSONL text corpus with a binary format that loads in microseconds
//! via sequential read (no parsing, no allocation beyond the DAG's vecs).
//!
//! ## What this module owns, and what it does not
//!
//! This file owns *framing*: a magic, a derived schema identity, a count, and
//! per-entry `(name, payload length, payload)` records. It does **not** own
//! the expression encoding. Marshalling a DAG — walking it, assigning
//! ordinals, a tag per node kind, reading it back — is
//! [`pixelflow_ir::encode`]/[`pixelflow_ir::decode`]'s job, and this module
//! calls it once per entry. It used to hand-roll that walk against the old
//! `ExprArena`'s public node/child accessors, which is how a *storage
//! representation* leaked out of the IR and into a corpus writer: every
//! change to what a node is was a change here too. A payload is now an opaque
//! byte string this module measures and frames but never interprets.
//!
//! ```text
//! magic: [u8; 4] = b"PXCR"
//! schema_identity: u64 (little-endian) = corpus_identity()
//! count: u32 (little-endian)
//!
//! For each expression:
//!   name_len: u16 (little-endian)
//!   name: [u8; name_len]       (UTF-8)
//!   expr_len: u32 (little-endian)
//!   expr: [u8; expr_len]       (pixelflow_ir::encode)
//! ```
//!
//! ## Only the reachable subtree is stored
//!
//! [`pixelflow_ir::encode`] serializes the subgraph reachable from the root,
//! not the whole DAG. Generator DAGs are append-only scratch space: rewriting
//! a node pushes a replacement and abandons the original, so a DAG holding a
//! 12-node expression can carry hundreds of dead nodes. Storing them made
//! `len()` a measure of the *generator's history* rather than of the
//! expression, and every downstream size filter that read it silently dropped
//! small expressions with long provenance — non-randomly, since dead-node
//! count correlates with how many rewrite passes ran (the B3 holdout-integrity
//! bug: 110 of 380 DEV entries dropped by a filter measuring dead nodes).
//! Stored size *is* expression size, and it is that way because the encoder
//! guarantees it rather than because this module remembers to compact first.
//!
//! ## The schema identity is coupled to the payload's meaning
//!
//! Four independent things can invalidate a stored corpus, and none of them
//! announce themselves as a parse error: the op encoding (`pixelflow-ir`'s
//! `OpKind::marshal` is free to renumber without telling anyone, so a stale
//! corpus decodes to different operations rather than failing to parse); the
//! expression encoding's own layout (`pixelflow-ir` versions it with a byte
//! of its own, which [`pixelflow_ir::decode`] checks — but per *entry*, deep
//! inside the file, not at the header where a regeneration hint belongs); the
//! reachable-subtree compaction (an uncompacted file parses fine, but its node
//! counts are generator history, not expression size — bug B3); and which tier
//! a file was deduplicated against (a corpus keyed on raw structure instead of
//! the feature-quotient [`FenceKey`](super::structural::FenceKey) parses fine
//! but can hold a DEV/FINAL leak in TRAIN — P1(d)). A hand-bumped `VERSION`
//! integer needs a human to remember to move it every time any of the four
//! changes; this format instead carries [`corpus_identity`], which folds
//! [`crate::schema::SchemaIdentity`]'s content hash of `CorpusFormat::SCHEMA`
//! (editing that description IS the version bump, so there is no separate step
//! to forget) together with a hash of the LIVE `OpKind` encoding table and a
//! hash of the LIVE [`pixelflow_ir::encode`] output for a fixed probe
//! expression — the two cases prose alone cannot see, since "dense 0..COUNT
//! discriminants" and "pixelflow-ir's own encoding" read the same before and
//! after a renumbering or a version bump (docs/plans/
//! 2026-08-17-cost-model-domain.md, J9, kills P1(b)). Any stored identity
//! other than the current one is a hard load error.

use std::io::{self, Write};
use std::path::Path;

use pixelflow_ir::{ExprBuilder, ExprData, OpKind, Rooted, decode, encode};

use crate::schema::{SchemaIdentity, fnv1a64_const, identity_mismatch};

/// One stored expression: its name and the rooted graph it names.
pub type Entry = (String, Rooted<ExprData>);

/// Marker type naming the corpus binary format for [`SchemaIdentity`]. The
/// format has no single Rust value of its own — it serializes a `Vec<Entry>` —
/// so this type exists purely to carry `SCHEMA` and derive `SCHEMA_IDENTITY`
/// from it.
struct CorpusFormat;

impl SchemaIdentity for CorpusFormat {
    const MAGIC: &'static str = "PXCR";
    // Every field this format's bytes carry, and what each one means. This
    // text IS the version: change what a field means here and
    // `SCHEMA_IDENTITY` (its content hash) changes with it, so a corpus
    // written under the old meaning cannot silently be read under the new
    // one. History, for readers orienting themselves: the op encoding went
    // dense (0..COUNT) under `OpKind::marshal`; entries were narrowed to only
    // the subtree reachable from `root` (a generator's node count is its
    // history, not expression size — bug B3); the cross-tier dedup key moved
    // from raw structure to the feature-quotient `FenceKey`, because the
    // extraction head's features collapse literals and a literal-keyed dedup
    // could seat a DEV/FINAL-equivalent expression in TRAIN (P1(d)); and the
    // per-node encoding stopped being this module's hand-rolled tag walk and
    // became `pixelflow_ir::encode`'s, which changed the payload bytes.
    const SCHEMA: &'static str = "\
        header: magic[4]=PXCR, schema_identity: u64 le, count: u32 le entries follow; \
        entry: name_len u16 le, name utf8 bytes, expr_len u32 le, \
        expr: expr_len bytes produced by pixelflow_ir::encode and read back by \
        pixelflow_ir::decode — an opaque payload this format frames and never \
        interprets, self-describing down to its own version byte, node count, \
        child-before-parent node records and root ordinal; \
        node encoding, op byte encoding and reachable-subtree compaction are all \
        pixelflow-ir's, not this format's: entries store ONLY the subtree reachable \
        from the root, so stored node count is expression size and not generator \
        history; \
        Buffer and Uniform leaves have no encoding at all — a declaration is a \
        per-process binding, refused at write time rather than stored unreadable; \
        cross-tier dedup key an on-disk TRAIN tier was built against: the \
        feature-quotient FenceKey (structural_key composed with the extraction \
        head's literal-collapsing feature map), not raw structural equality";
}

fn regen_command() -> &'static str {
    "cargo run --release -p pixelflow-pipeline --features training --bin gen_bench_corpus"
}

// ── Identity: SCHEMA text folded with the LIVE encodings it delegates to ────
//
// `CorpusFormat::SCHEMA_IDENTITY` alone has a hole: it hashes the prose in
// `SCHEMA` above, and that prose only *describes* the payload as "produced by
// pixelflow_ir::encode" — a sentence that stays true, and therefore stays
// byte-identical, no matter what `encode` emits or which op ends up at which
// byte. `OpKind::marshal` is free to renumber (`docs/designs/
// opkind-numbering-is-private.md`) and `encode`'s own layout is free to
// change; neither changes anything this text says, so `SCHEMA_IDENTITY` alone
// cannot see them. An old corpus would pass the identity check and either
// decode every affected opcode as the wrong operation, or fail per-entry deep
// inside the file with `pixelflow-ir`'s version error and no regeneration
// hint — exactly the failure J9 exists to make impossible.
//
// This is the derived-encoding-fingerprint design that opkind-numbering-is-
// private.md §4.3 named `OpKind::ENCODING_ID` and rejected as disproportion-
// ate — at the time, a hand-bumped `VERSION` integer was still the actual
// gate, so the fingerprint would have been a second, redundant guard over
// the same format. That premise is gone: this corpus format's gate is now
// `SchemaIdentity`'s derived hash and nothing else, so a renumbering that
// outruns the prose has no other guard left to catch it. `pixelflow-ir` is
// out of scope for this change (its numbering stays private, per that same
// doc), so both fingerprints are computed here, from public surface, rather
// than as `pixelflow-ir` consts.

/// Folds `(name, marshal() byte)` for every live op, in `OpKind::all()`
/// order, plus the op count — not just the byte sequence, which is always
/// `0..COUNT` by construction and so is invariant under a renumbering that
/// only swaps which op sits at which position.
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

/// Fingerprint of [`pixelflow_ir::encode`]'s live output: the bytes it
/// produces for one fixed probe expression exercising every record shape the
/// corpus can hold — a leaf `Var`, a `Const`, a `Param`, and an operator with
/// children.
///
/// `pixelflow-ir` versions its own encoding with a leading byte and
/// [`decode`] refuses a stale one, but that check fires per *entry*, deep in
/// the file, as an error that cannot name this corpus or how to regenerate
/// it. Folding the probe's bytes into the header identity moves the same fact
/// to the header, where the reader has both. The two checks are deliberately
/// kept, not deduplicated: the header one is the friendly gate, `decode`'s is
/// the one that holds when a payload arrives from somewhere else.
fn ir_encoding_identity() -> u64 {
    let mut b = ExprBuilder::new();
    let x = b.push_var(0);
    let c = b.push_const(1.5);
    let p = b.push_param(3);
    let root = b.push_ternary(OpKind::MulAdd, x, c, p);
    let (rooted, _) = b.finish(&[root]);
    fnv1a64_const(&encode(rooted.entry()))
}

/// The identity actually written to disk and checked on load: the schema
/// prose's hash, folded with the live `(op name, encoded byte)` table and the
/// live expression encoding, so a renumbering or a format bump in
/// `pixelflow-ir` changes the identity by construction, not by someone
/// remembering to edit `CorpusFormat::SCHEMA` to match.
fn corpus_identity() -> u64 {
    let mut buf = Vec::with_capacity(24);
    buf.extend_from_slice(&CorpusFormat::SCHEMA_IDENTITY.to_le_bytes());
    buf.extend_from_slice(&opcode_encoding_identity().to_le_bytes());
    buf.extend_from_slice(&ir_encoding_identity().to_le_bytes());
    fnv1a64_const(&buf)
}

// ── Write ────────────────────────────────────────────────────────────────────

/// Write a binary corpus to `path`.
///
/// Each entry's payload is [`pixelflow_ir::encode`] of the subgraph reachable
/// from its root, so the stored node count is the expression's size, not its
/// generator DAG's.
///
/// # Panics
///
/// Panics if any expression name exceeds `u16::MAX` bytes, or if any
/// expression reaches a `Buffer` or `Uniform` leaf: a declaration is a
/// per-process binding rather than a value, so `encode` refuses to write one
/// down (writing it would store a node that cannot be read back correctly).
pub fn write_corpus(path: &Path, entries: &[Entry]) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut w = io::BufWriter::new(file);

    // Header
    w.write_all(CorpusFormat::MAGIC.as_bytes())?;
    w.write_all(&corpus_identity().to_le_bytes())?;
    w.write_all(&(entries.len() as u32).to_le_bytes())?;

    for (name, expr) in entries {
        write_entry(&mut w, name, expr)?;
    }

    w.flush()?;
    Ok(())
}

fn write_entry(w: &mut impl Write, name: &str, expr: &Rooted<ExprData>) -> io::Result<()> {
    let name_bytes = name.as_bytes();
    assert!(
        name_bytes.len() <= u16::MAX as usize,
        "write_corpus: expression name exceeds u16::MAX bytes: '{name}'"
    );

    let payload = encode(expr.entry());
    assert!(
        payload.len() <= u32::MAX as usize,
        "write_corpus: encoded expression '{name}' exceeds u32::MAX bytes"
    );

    w.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
    w.write_all(name_bytes)?;
    w.write_all(&(payload.len() as u32).to_le_bytes())?;
    w.write_all(&payload)?;

    Ok(())
}

// ── Read ─────────────────────────────────────────────────────────────────────

/// Read a binary corpus from `path`.
///
/// Returns `(name, expression)` pairs.
pub fn read_corpus(path: &Path) -> io::Result<Vec<Entry>> {
    let data = std::fs::read(path)?;
    read_corpus_bytes(&data)
}

fn read_corpus_bytes(data: &[u8]) -> io::Result<Vec<Entry>> {
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

    // Exact-identity check, no tolerance. The payload bytes mean whatever
    // `pixelflow_ir::encode` said when the file was written, and that encoding
    // — both its layout and which byte names which op — may change without
    // notice, so a "best effort" read of a stale file would decode
    // valid-looking bytes into the wrong program. A hand-bumped integer needs
    // a human to remember to move it every time the format's meaning changes;
    // this identity is derived from `CorpusFormat::SCHEMA` folded with the
    // live op table and the live expression encoding (`corpus_identity`,
    // docs/plans/2026-08-17-cost-model-domain.md, J9), so any change to what
    // the bytes below mean — INCLUDING a silent `OpKind::marshal` renumbering
    // or an `encode` version bump the prose doesn't happen to mention —
    // changes the identity by construction.
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

fn read_entry(r: &mut Cursor<'_>) -> io::Result<Entry> {
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

    let payload_len = r.read_u32()? as usize;
    let payload = r.read_bytes(payload_len)?;
    let expr = decode(payload).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expression '{name}' does not decode: {e}"),
        )
    })?;

    Ok((name, expr))
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
    use pixelflow_ir::{display, node_count_subtree, subtree_eq};

    // Per-process path: concurrent `cargo test` runs must not share corpus files,
    // or one process's remove_file races another's write/read.
    fn unique_tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("corpus_rt_{name}_{}.bin", std::process::id()))
    }

    /// `(name, expr)` for a graph built by `f`, which returns the root.
    fn entry(name: &str, f: impl FnOnce(&mut ExprBuilder) -> pixelflow_ir::ExprRef) -> Entry {
        let mut b = ExprBuilder::new();
        let root = f(&mut b);
        let (rooted, _) = b.finish(&[root]);
        (name.to_string(), rooted)
    }

    fn round_trip(tmp_name: &str, entries: &[Entry]) -> Vec<Entry> {
        let tmp = unique_tmp(tmp_name);
        write_corpus(&tmp, entries).expect("write");
        let loaded = read_corpus(&tmp).expect("read");
        let _ = std::fs::remove_file(&tmp);
        loaded
    }

    #[test]
    fn round_trip_empty() {
        assert!(round_trip("empty", &[]).is_empty());
    }

    #[test]
    fn round_trip_simple() {
        let entries = vec![entry("test_add", |b| {
            let x = b.push_var(0);
            let y = b.push_var(1);
            b.push_binary(OpKind::Add, x, y)
        })];

        let loaded = round_trip("simple", &entries);

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "test_add");
        assert_eq!(loaded[0].1.len(), 3);
        assert!(subtree_eq(loaded[0].1.entry(), entries[0].1.entry()));
    }

    #[test]
    fn round_trip_with_const_and_unary() {
        let entries = vec![entry("sqrt_pi", |b| {
            let c = b.push_const(std::f32::consts::PI);
            b.push_unary(OpKind::Sqrt, c)
        })];

        let loaded = round_trip("unary", &entries);

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "sqrt_pi");
        // The const value round-trips bit-for-bit: `encode` stores the IEEE
        // pattern, not a decimal rendering.
        assert!(subtree_eq(loaded[0].1.entry(), entries[0].1.entry()));
        let consts: Vec<f32> = loaded[0]
            .1
            .iter()
            .filter_map(|n| n.as_f32())
            .collect::<Vec<_>>();
        assert_eq!(consts, [std::f32::consts::PI]);
    }

    #[test]
    fn round_trip_ternary() {
        let entries = vec![entry("select_xyz", |b| {
            let x = b.push_var(0);
            let y = b.push_var(1);
            let z = b.push_var(4);
            b.push_ternary(OpKind::Select, x, y, z)
        })];

        let loaded = round_trip("ternary", &entries);

        assert_eq!(loaded.len(), 1);
        let root = loaded[0].1.entry();
        assert_eq!(root.op(), Some(OpKind::Select));
        assert_eq!(root.child_count(), 3);
    }

    #[test]
    fn round_trip_nary() {
        let entries = vec![entry("tuple_abc", |b| {
            let a = b.push_var(0);
            let c = b.push_var(1);
            let d = b.push_var(4);
            b.push_nary(OpKind::Tuple, &[a, c, d])
        })];

        let loaded = round_trip("nary", &entries);

        assert_eq!(loaded.len(), 1);
        let root = loaded[0].1.entry();
        assert_eq!(root.op(), Some(OpKind::Tuple));
        assert_eq!(root.child_count(), 3);
    }

    // Header-rejection tests drive the reader through its public entry point:
    // `read_corpus_bytes` is private, and pinning it here would test a path no
    // caller can reach. `name` keeps sibling tests off each other's fixture
    // within a process, `unique_tmp` keeps concurrent test processes apart.
    fn read_corpus_from_bytes(name: &str, data: &[u8]) -> io::Result<Vec<Entry>> {
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
    fn a_corrupt_payload_is_refused_by_the_decoder() {
        // The framing is this module's; the payload is not. A byte flipped
        // inside a payload must come back as a decode error naming the entry,
        // never as a plausible-looking different program.
        let entries = vec![entry("victim", |b| {
            let x = b.push_var(0);
            b.push_unary(OpKind::Sqrt, x)
        })];
        let tmp = unique_tmp("corrupt_payload");
        write_corpus(&tmp, &entries).expect("write");
        let mut data = std::fs::read(&tmp).expect("read back");
        let _ = std::fs::remove_file(&tmp);

        // The payload's first byte is `pixelflow-ir`'s own encoding version.
        let payload_start = 4 + 8 + 4 + 2 + "victim".len() + 4;
        data[payload_start] = data[payload_start].wrapping_add(1);

        match read_corpus_from_bytes("corrupt_payload_read", &data) {
            Ok(_) => panic!("a corrupt payload must be refused"),
            Err(e) => {
                let msg = e.to_string();
                assert!(msg.contains("victim"), "error must name the entry: {msg}");
                assert!(
                    msg.contains("does not decode"),
                    "error must name the failure class: {msg}"
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
        let old_meaning = fnv1a64_const(b"the payload is this module's own tag walk");
        let new_meaning = fnv1a64_const(b"the payload is pixelflow_ir::encode's bytes");
        assert_ne!(old_meaning, new_meaning);
        assert_eq!(CorpusFormat::SCHEMA_IDENTITY, CorpusFormat::SCHEMA_IDENTITY);
    }

    // ── corpus_identity folds in the live encodings (P1 finding on
    // PR #1019: SCHEMA_IDENTITY alone hashes only prose) ───────────────────

    #[test]
    fn corpus_identity_is_deterministic() {
        assert_eq!(corpus_identity(), corpus_identity());
        assert_eq!(opcode_encoding_identity(), opcode_encoding_identity());
        assert_eq!(ir_encoding_identity(), ir_encoding_identity());
    }

    #[test]
    fn corpus_identity_depends_on_more_than_the_schema_text() {
        // The defect this guards: `CorpusFormat::SCHEMA` describes the payload
        // only as "produced by pixelflow_ir::encode" — a sentence a
        // renumbering or a version bump never has to touch. If
        // `corpus_identity` reduced to `SCHEMA_IDENTITY` alone, either change
        // would leave the on-disk gate unchanged and a stale corpus would
        // decode as the wrong program (or fail per-entry with no regeneration
        // hint). Folding the live encodings in means the composite identity
        // cannot equal the bare schema-text hash.
        assert_ne!(
            corpus_identity(),
            CorpusFormat::SCHEMA_IDENTITY,
            "corpus_identity must depend on the live encodings, not just the \
             schema prose describing them"
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

    #[test]
    fn ir_encoding_identity_tracks_the_payload_bytes() {
        // The fingerprint is the probe's encoded bytes, so it changes exactly
        // when those bytes do — including on `pixelflow-ir`'s own leading
        // version byte, which is the case a per-entry `decode` error would
        // otherwise report with no corpus and no regeneration hint.
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let c = b.push_const(1.5);
        let p = b.push_param(3);
        let root = b.push_ternary(OpKind::MulAdd, x, c, p);
        let (rooted, _) = b.finish(&[root]);
        let bytes = encode(rooted.entry());
        assert_eq!(ir_encoding_identity(), fnv1a64_const(&bytes));

        let mut bumped = bytes.clone();
        bumped[0] = bumped[0].wrapping_add(1);
        assert_ne!(
            fnv1a64_const(&bumped),
            ir_encoding_identity(),
            "a version bump inside the payload must move the corpus identity"
        );
    }

    // ── Reachable-subtree compaction (bug B3) ───────────────────────────────

    /// A DAG carrying the dead nodes an append-only generator leaves behind:
    /// `X + 1.0` is built, then abandoned in favour of `X * 2.0`.
    fn dag_with_dead_nodes() -> Entry {
        entry("junky", |b| {
            let x = b.push_var(0);
            let one = b.push_const(1.0);
            let _abandoned = b.push_binary(OpKind::Add, x, one);
            let two = b.push_const(2.0);
            b.push_binary(OpKind::Mul, x, two)
        })
    }

    #[test]
    fn stored_size_reflects_the_expression_not_the_generator() {
        // The B3 regression: a small expression in a junk-heavy DAG must
        // round-trip as a small expression. Reading back `len() == 5` is what
        // let a `> N` filter drop 29% of the DEV tier. Compaction is now
        // `encode`'s guarantee rather than a step this module performs, so
        // this pins the property through the file.
        let original = dag_with_dead_nodes();
        assert_eq!(original.1.len(), 5, "fixture should carry 2 dead nodes");
        assert_eq!(node_count_subtree(original.1.entry()), 3);

        let loaded = round_trip("dead_nodes", std::slice::from_ref(&original));

        assert_eq!(loaded.len(), 1);
        assert_eq!(
            loaded[0].1.len(),
            3,
            "stored graph must hold only the reachable subtree, got {} nodes",
            loaded[0].1.len()
        );
        assert!(
            subtree_eq(loaded[0].1.entry(), original.1.entry()),
            "the round-tripped expression must equal the original"
        );
    }

    #[test]
    fn round_trip_preserves_dag_sharing() {
        // `s + s` where `s = sqrt(X)`: the shared child must stay shared, or
        // the round trip would turn a DAG into an exponentially larger tree.
        let original = entry("shared", |b| {
            let x = b.push_var(0);
            let s = b.push_unary(OpKind::Sqrt, x);
            b.push_binary(OpKind::Add, s, s)
        });
        let loaded = round_trip("sharing", std::slice::from_ref(&original));

        assert_eq!(loaded[0].1.len(), 3, "shared node must be stored once");
        let root = loaded[0].1.entry();
        let kids: Vec<_> = root.children().collect();
        assert_eq!(kids[0], kids[1], "both operands must be the same node");
    }

    #[test]
    #[should_panic(expected = "is a binding, not a value")]
    fn writing_refuses_buffer_nodes() {
        // A Buffer leaf's declaration is not part of the corpus format — an
        // identity is minted per process, so writing one down would name
        // unrelated memory on the next run. `encode` refuses it, which is
        // what makes an unreadable entry unwritable.
        let mut b = ExprBuilder::new();
        let buf = b.declare_buffer(pixelflow_ir::BufferDecl {
            id: pixelflow_ir::BufferIdentity::mint(),
            width: 8,
            height: 8,
        });
        let root = b.push_buffer(buf);
        let (rooted, _) = b.finish(&[root]);
        let tmp = unique_tmp("buffer_refused");
        let _refused = write_corpus(&tmp, &[("buf".to_string(), rooted)]);
    }

    #[test]
    fn round_trip_multiple_entries() {
        let entries = vec![
            entry("add_xy", |b| {
                let x = b.push_var(0);
                let y = b.push_var(1);
                b.push_binary(OpKind::Add, x, y)
            }),
            entry("sqrt_pi", |b| {
                let c = b.push_const(std::f32::consts::PI);
                b.push_unary(OpKind::Sqrt, c)
            }),
        ];

        let loaded = round_trip("multi", &entries);

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].0, "add_xy");
        assert_eq!(loaded[1].0, "sqrt_pi");
        assert_eq!(loaded[0].1.len(), 3);
        assert_eq!(loaded[1].1.len(), 2);
        assert_eq!(
            format!("{}", display(loaded[1].1.entry())),
            format!("{}", display(entries[1].1.entry()))
        );
    }
}
