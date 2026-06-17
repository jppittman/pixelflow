//! Persistent file-backed replay buffer for training.
//!
//! Fixed-size `#[repr(C)]` records are written flat to a binary file — no
//! serialization overhead. On open, existing records are loaded from disk so
//! training can resume after a crash.
//!
//! File format:
//! - Bytes 0..8:  record count as little-endian u64
//! - Bytes 8..:   N × [`ReplayRecord`], contiguous

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// INPUT_DIM from ExprNnue (4*K + 2 = 130).
const ACC_DIM: usize = 130;
/// EMBED_DIM from ExprNnue.
const EMBED_DIM: usize = 24;

const HEADER_SIZE: usize = 8;
const RECORD_SIZE: usize = std::mem::size_of::<ReplayRecord>();

// Compile-time size check — the layout must be exactly 640 bytes for
// stable on-disk format.
const _: () = assert!(RECORD_SIZE == 640);

/// Fixed-size record for the persistent replay buffer.
///
/// `#[repr(C)]` with only primitive fields — safe to transmute to/from bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReplayRecord {
    pub acc: [f32; ACC_DIM],          // 130 × 4 = 520
    pub rule_embed: [f32; EMBED_DIM], // 24 × 4 = 96
    pub rule_idx: u32,                // 4
    pub matched: u32,                 // bool as u32 for alignment, 4
    pub jit_cost_ns: f64,             // 8
    pub advantage: f32,               // 4
    pub _pad: f32,                    // 4  → total 640
}

/// Persistent replay buffer backed by a flat binary file.
///
/// Records live in a `Vec` for fast random sampling and are flushed to disk
/// on [`MmapReplayBuffer::sync`].
pub struct MmapReplayBuffer {
    path: PathBuf,
    file: File,
    records: Vec<ReplayRecord>,
    max_records: usize,
}

impl MmapReplayBuffer {
    /// Open or create a replay buffer file. Loads existing records if present.
    ///
    /// # Panics
    /// Panics on any I/O error (no silent failures).
    pub fn open(path: &Path, max_records: usize) -> Self {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .unwrap_or_else(|e| panic!("[REPLAY] Failed to open {:?}: {e}", path));

        let metadata = file
            .metadata()
            .unwrap_or_else(|e| panic!("[REPLAY] Failed to stat {:?}: {e}", path));
        let file_len = metadata.len() as usize;

        let records = if file_len >= HEADER_SIZE {
            Self::load_records(&file, file_len, path)
        } else {
            if file_len > 0 {
                eprintln!(
                    "[REPLAY] File {:?} too small ({file_len} bytes) for header, starting fresh",
                    path,
                );
            }
            eprintln!("[REPLAY] Created new replay buffer at {}", path.display());
            Vec::new()
        };

        if !records.is_empty() {
            eprintln!(
                "[REPLAY] Loaded {} records from {}",
                records.len(),
                path.display(),
            );
        }

        Self {
            path: path.to_path_buf(),
            file,
            records,
            max_records,
        }
    }

    fn load_records(mut file: &File, file_len: usize, path: &Path) -> Vec<ReplayRecord> {
        let mut header = [0u8; HEADER_SIZE];
        (&mut file)
            .read_exact(&mut header)
            .unwrap_or_else(|e| panic!("[REPLAY] Failed to read header from {:?}: {e}", path));
        let count = u64::from_le_bytes(header) as usize;

        let expected = HEADER_SIZE + count * RECORD_SIZE;
        if file_len < expected {
            panic!(
                "[REPLAY] File {:?} claims {count} records ({expected} bytes needed) \
                 but is only {file_len} bytes — file is corrupt",
                path,
            );
        }

        let mut buf = vec![0u8; count * RECORD_SIZE];
        (&mut file)
            .read_exact(&mut buf)
            .unwrap_or_else(|e| panic!("[REPLAY] Failed to read records from {:?}: {e}", path));

        bytes_as_records(&buf).to_vec()
    }

    /// Number of records in buffer.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Push a single record. If over capacity, FIFO evict oldest.
    pub fn push(&mut self, record: ReplayRecord) {
        self.records.push(record);
        if self.records.len() > self.max_records {
            let excess = self.records.len() - self.max_records;
            self.records.drain(..excess);
        }
    }

    /// Flush all records to disk. Call after each round.
    ///
    /// # Panics
    /// Panics on any I/O error.
    pub fn sync(&mut self) {
        self.file
            .seek(SeekFrom::Start(0))
            .unwrap_or_else(|e| panic!("[REPLAY] Failed to seek in {:?}: {e}", self.path));

        let count = self.records.len() as u64;
        self.file
            .write_all(&count.to_le_bytes())
            .unwrap_or_else(|e| panic!("[REPLAY] Failed to write header to {:?}: {e}", self.path));

        self.file
            .write_all(records_as_bytes(&self.records))
            .unwrap_or_else(|e| panic!("[REPLAY] Failed to write records to {:?}: {e}", self.path));

        // Truncate in case the file previously held more records.
        let new_len = (HEADER_SIZE + self.records.len() * RECORD_SIZE) as u64;
        self.file
            .set_len(new_len)
            .unwrap_or_else(|e| panic!("[REPLAY] Failed to truncate {:?}: {e}", self.path));

        self.file
            .sync_all()
            .unwrap_or_else(|e| panic!("[REPLAY] Failed to fsync {:?}: {e}", self.path));
    }

    /// Get record by index.
    ///
    /// # Panics
    /// Panics if `idx >= len()`.
    pub fn get(&self, idx: usize) -> &ReplayRecord {
        assert!(
            idx < self.records.len(),
            "[REPLAY] Index {idx} out of bounds (len={})",
            self.records.len(),
        );
        &self.records[idx]
    }

    /// Sample `batch_size` random indices using LCG PRNG.
    ///
    /// # Panics
    /// Panics if the buffer is empty.
    pub fn sample_batch(&self, batch_size: usize, seed: u64) -> Vec<usize> {
        let n = self.records.len();
        assert!(n > 0, "[REPLAY] Cannot sample from empty replay buffer");
        let batch_size = batch_size.min(n);
        let mut indices = Vec::with_capacity(batch_size);
        let mut state = seed;
        for _ in 0..batch_size {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            indices.push((state >> 33) as usize % n);
        }
        indices
    }
}

// ── Raw byte helpers ─────────────────────────────────────────────────────

fn records_as_bytes(records: &[ReplayRecord]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(records.as_ptr() as *const u8, records.len() * RECORD_SIZE)
    }
}

fn bytes_as_records(bytes: &[u8]) -> &[ReplayRecord] {
    assert!(
        bytes.len().is_multiple_of(RECORD_SIZE),
        "[REPLAY] Byte buffer length {} is not a multiple of record size {RECORD_SIZE}",
        bytes.len(),
    );
    assert!(
        (bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<ReplayRecord>())
            || bytes.is_empty(),
        "[REPLAY] Byte buffer is not aligned to ReplayRecord alignment",
    );
    unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const ReplayRecord,
            bytes.len() / RECORD_SIZE,
        )
    }
}
