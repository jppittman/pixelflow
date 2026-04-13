//! Binary corpus format for pre-parsed expression storage.
//!
//! Replaces JSONL text corpus with a binary format that loads in microseconds
//! via sequential read (no parsing, no allocation beyond the arena vecs).
//!
//! ## Format (v1)
//!
//! ```text
//! magic: [u8; 4] = b"PXCR"
//! version: u32 (little-endian) = 1
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

use std::io::{self, Write};
use std::path::Path;

use pixelflow_ir::{ExprArena, ExprId, ExprNode, OpKind};

const MAGIC: &[u8; 4] = b"PXCR";
const VERSION: u32 = 1;

// ── ExprNode serialization tags ──────────────────────────────────────────────

const TAG_VAR: u8 = 0;
const TAG_CONST: u8 = 1;
const TAG_PARAM: u8 = 2;
const TAG_UNARY: u8 = 3;
const TAG_BINARY: u8 = 4;
const TAG_TERNARY: u8 = 5;
const TAG_NARY: u8 = 6;

// ── Write ────────────────────────────────────────────────────────────────────

/// Write a binary corpus to `path`.
///
/// # Panics
///
/// Panics if any expression name exceeds `u16::MAX` bytes.
pub fn write_corpus(path: &Path, entries: &[(String, ExprArena, ExprId)]) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut w = io::BufWriter::new(file);

    // Header
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
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

    let nodes = arena.nodes_raw();
    let nary = arena.nary_children_raw();

    w.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
    w.write_all(name_bytes)?;
    w.write_all(&(nodes.len() as u32).to_le_bytes())?;
    w.write_all(&(nary.len() as u32).to_le_bytes())?;
    w.write_all(&root.0.to_le_bytes())?;

    // Nodes
    for node in nodes {
        write_node(w, node)?;
    }

    // Nary children
    for child in nary {
        w.write_all(&child.0.to_le_bytes())?;
    }

    Ok(())
}

fn write_node(w: &mut impl Write, node: &ExprNode) -> io::Result<()> {
    match node {
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
            w.write_all(&[TAG_UNARY, *op as u8])?;
            w.write_all(&a.0.to_le_bytes())?;
        }
        ExprNode::Binary(op, a, b) => {
            w.write_all(&[TAG_BINARY, *op as u8])?;
            w.write_all(&a.0.to_le_bytes())?;
            w.write_all(&b.0.to_le_bytes())?;
        }
        ExprNode::Ternary(op, a, b, c) => {
            w.write_all(&[TAG_TERNARY, *op as u8])?;
            w.write_all(&a.0.to_le_bytes())?;
            w.write_all(&b.0.to_le_bytes())?;
            w.write_all(&c.0.to_le_bytes())?;
        }
        ExprNode::Nary(op, start, len) => {
            w.write_all(&[TAG_NARY, *op as u8])?;
            w.write_all(&start.to_le_bytes())?;
            w.write_all(&len.to_le_bytes())?;
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
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad corpus magic: expected {:?}, got {:?}", MAGIC, magic),
        ));
    }

    let version = r.read_u32()?;
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported corpus version: {version} (expected {VERSION})"),
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
    let nary_count = r.read_u32()? as usize;
    let root_index = r.read_u32()?;

    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        nodes.push(read_node(r)?);
    }

    let mut nary_children = Vec::with_capacity(nary_count);
    for _ in 0..nary_count {
        nary_children.push(ExprId(r.read_u32()?));
    }

    let arena = ExprArena::from_raw(nodes, nary_children);
    let root = ExprId(root_index);

    Ok((name, arena, root))
}

fn read_node(r: &mut Cursor<'_>) -> io::Result<ExprNode> {
    let tag = r.read_u8()?;
    match tag {
        TAG_VAR => {
            let i = r.read_u8()?;
            Ok(ExprNode::Var(i))
        }
        TAG_CONST => {
            let bits = r.read_u32()?;
            Ok(ExprNode::Const(f32::from_le_bytes(bits.to_le_bytes())))
        }
        TAG_PARAM => {
            let i = r.read_u8()?;
            Ok(ExprNode::Param(i))
        }
        TAG_UNARY => {
            let op = read_opkind(r)?;
            let a = ExprId(r.read_u32()?);
            Ok(ExprNode::Unary(op, a))
        }
        TAG_BINARY => {
            let op = read_opkind(r)?;
            let a = ExprId(r.read_u32()?);
            let b = ExprId(r.read_u32()?);
            Ok(ExprNode::Binary(op, a, b))
        }
        TAG_TERNARY => {
            let op = read_opkind(r)?;
            let a = ExprId(r.read_u32()?);
            let b = ExprId(r.read_u32()?);
            let c = ExprId(r.read_u32()?);
            Ok(ExprNode::Ternary(op, a, b, c))
        }
        TAG_NARY => {
            let op = read_opkind(r)?;
            let start = r.read_u32()?;
            let len = r.read_u16()?;
            Ok(ExprNode::Nary(op, start, len))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown ExprNode tag: {tag}"),
        )),
    }
}

fn read_opkind(r: &mut Cursor<'_>) -> io::Result<OpKind> {
    let idx = r.read_u8()?;
    OpKind::from_index(idx as usize).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid OpKind index: {idx}"),
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
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_empty() {
        let tmp = std::env::temp_dir().join("corpus_rt_empty.bin");
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

        let tmp = std::env::temp_dir().join("corpus_rt_simple.bin");
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
        let c = arena.push_const(3.14);
        let root = arena.push_unary(OpKind::Sqrt, c);

        let entries = vec![("sqrt_pi".to_string(), arena, root)];

        let tmp = std::env::temp_dir().join("corpus_rt_unary.bin");
        write_corpus(&tmp, &entries).expect("write");
        let loaded = read_corpus(&tmp).expect("read");

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "sqrt_pi");
        // Check the const value round-trips
        match loaded[0].1.node(ExprId(0)) {
            ExprNode::Const(v) => assert!((v - 3.14).abs() < 1e-6, "const mismatch: {v}"),
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

        let tmp = std::env::temp_dir().join("corpus_rt_ternary.bin");
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

        let tmp = std::env::temp_dir().join("corpus_rt_nary.bin");
        write_corpus(&tmp, &entries).expect("write");
        let loaded = read_corpus(&tmp).expect("read");

        assert_eq!(loaded.len(), 1);
        match loaded[0].1.node(loaded[0].2) {
            ExprNode::Nary(OpKind::Tuple, start, len) => {
                assert_eq!(*len, 3);
                let children = loaded[0].1.nary_children_slice(*start, *len);
                assert_eq!(children.len(), 3);
            }
            other => panic!("expected Nary(Tuple,...), got {other:?}"),
        }

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn bad_magic_fails() {
        let data = b"BADMxxxxxxxx";
        match read_corpus_bytes(data) {
            Ok(_) => panic!("expected error for bad magic"),
            Err(e) => assert!(
                e.to_string().contains("bad corpus magic"),
                "unexpected error: {e}"
            ),
        }
    }

    #[test]
    fn bad_version_fails() {
        let mut data = Vec::new();
        data.extend_from_slice(MAGIC);
        data.extend_from_slice(&99u32.to_le_bytes()); // bad version
        data.extend_from_slice(&0u32.to_le_bytes()); // count=0
        match read_corpus_bytes(&data) {
            Ok(_) => panic!("expected error for bad version"),
            Err(e) => assert!(
                e.to_string().contains("unsupported corpus version"),
                "unexpected error: {e}"
            ),
        }
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

        // Entry 2: sqrt(3.14)
        let mut a2 = ExprArena::new();
        let c = a2.push_const(3.14);
        let r2 = a2.push_unary(OpKind::Sqrt, c);
        entries.push(("sqrt_pi".to_string(), a2, r2));

        let tmp = std::env::temp_dir().join("corpus_rt_multi.bin");
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
