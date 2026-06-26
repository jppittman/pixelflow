//! Generic font loading abstraction.
//!
//! This module provides a trait-based approach to loading font data from
//! different sources:
//!
//! - `DataSource` - Loads from owned byte vectors
//! - `EmbeddedSource` - Uses fonts compiled into the binary
//! - `MmapSource` - Memory-maps font files for zero-copy access
//!
//! # Example
//!
//! ```ignore
//! use pixelflow_graphics::fonts::loader::{FontSource, MmapSource, LoadedFont};
//!
//! // Load a large font via mmap (doesn't bloat binary)
//! let source = MmapSource::open("/path/to/font.ttf")?;
//! let loaded = LoadedFont::new(source)?;
//! let glyph = loaded.font().glyph_scaled('A', 16.0);
//! ```

use super::ttf::Font;
use std::fs::File;
use std::io;
use std::path::Path;

#[cfg(unix)]
use std::os::unix::io::AsRawFd;

// ═══════════════════════════════════════════════════════════════════════════
// FontSource Trait
// ═══════════════════════════════════════════════════════════════════════════

/// A source of font data.
///
/// This trait abstracts over different ways to provide font bytes:
/// - In-memory data (`DataSource`)
/// - Compiled-in static data (`EmbeddedSource`)
/// - Memory-mapped files (`MmapSource`)
///
/// All implementations must provide a contiguous byte slice that remains
/// valid for the lifetime of the source.
pub trait FontSource {
    /// Returns the font data as a byte slice.
    fn as_bytes(&self) -> &[u8];
}

// ═══════════════════════════════════════════════════════════════════════════
// DataSource - Owned byte vector
// ═══════════════════════════════════════════════════════════════════════════

/// Font source backed by an owned byte vector.
///
/// Use this when you have font data loaded dynamically (e.g., from network
/// or generated programmatically).
///
/// # Example
///
/// ```ignore
/// let data = std::fs::read("font.ttf")?;
/// let source = DataSource::new(data);
/// let loaded = LoadedFont::new(source)?;
/// ```
#[derive(Clone)]
pub struct DataSource {
    data: Vec<u8>,
}

impl DataSource {
    /// Create a new data source from a byte vector.
    #[must_use]
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Load font data from a file into memory.
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        Ok(Self::new(data))
    }
}

impl FontSource for DataSource {
    fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EmbeddedSource - Static compiled-in data
// ═══════════════════════════════════════════════════════════════════════════

/// Font source backed by static data compiled into the binary.
///
/// Use this with `include_bytes!()` to embed fonts at compile time.
/// The font data lives in the binary's read-only data section.
///
/// # Example
///
/// ```ignore
/// static FONT_DATA: &[u8] = include_bytes!("../assets/font.ttf");
/// let source = EmbeddedSource::new(FONT_DATA);
/// let loaded = LoadedFont::new(source)?;
/// ```
#[derive(Clone, Copy)]
pub struct EmbeddedSource {
    data: &'static [u8],
}

impl EmbeddedSource {
    /// Create a new embedded source from static data.
    ///
    /// Typically used with `include_bytes!()`.
    #[must_use]
    pub const fn new(data: &'static [u8]) -> Self {
        Self { data }
    }
}

impl FontSource for EmbeddedSource {
    fn as_bytes(&self) -> &[u8] {
        self.data
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MmapSource - Memory-mapped file
// ═══════════════════════════════════════════════════════════════════════════

/// Font source backed by a memory-mapped file.
///
/// Memory mapping allows loading large fonts without copying data into
/// the process's heap. The OS lazily pages in font data as needed,
/// making this ideal for large fonts (CJK, emoji, etc.) that would
/// otherwise bloat the binary or consume excessive memory.
///
/// # Platform Support
///
/// - **Unix** (Linux, macOS): Uses `mmap(2)`
/// - **Windows**: Uses `CreateFileMapping`/`MapViewOfFile`
///
/// # Safety
///
/// The mapped file must not be modified or truncated while in use.
/// This is enforced by opening the file in read-only mode.
///
/// # Example
///
/// ```ignore
/// let source = MmapSource::open("/usr/share/fonts/NotoSans.ttf")?;
/// let loaded = LoadedFont::new(source)?;
/// // Font data is paged in on-demand by the OS
/// ```
pub struct MmapSource {
    #[cfg(unix)]
    mmap: MmapInner,
    #[cfg(windows)]
    mmap: MmapInner,
}

#[cfg(unix)]
struct MmapInner {
    ptr: *mut u8,
    len: usize,
}

#[cfg(unix)]
impl MmapInner {
    fn new(file: &File) -> io::Result<Self> {
        use std::ptr;

        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cannot mmap empty file",
            ));
        }

        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        Ok(Self {
            ptr: ptr as *mut u8,
            len,
        })
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

#[cfg(unix)]
impl Drop for MmapInner {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

// SAFETY: The mmap is read-only and the underlying file descriptor is not
// accessible, so the memory region is effectively immutable.
#[cfg(unix)]
unsafe impl Send for MmapInner {}
#[cfg(unix)]
unsafe impl Sync for MmapInner {}

#[cfg(windows)]
struct MmapInner {
    ptr: *mut u8,
    len: usize,
    mapping: std::os::windows::io::RawHandle,
}

#[cfg(windows)]
impl MmapInner {
    fn new(file: &File) -> io::Result<Self> {
        use std::os::windows::io::AsRawHandle;
        use std::ptr;

        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cannot mmap empty file",
            ));
        }

        // Create file mapping
        let mapping = unsafe {
            windows_sys::Win32::System::Memory::CreateFileMappingW(
                file.as_raw_handle() as isize,
                ptr::null(),
                windows_sys::Win32::System::Memory::PAGE_READONLY,
                0,
                0,
                ptr::null(),
            )
        };

        if mapping == 0 {
            return Err(io::Error::last_os_error());
        }

        // Map view of file
        let ptr = unsafe {
            windows_sys::Win32::System::Memory::MapViewOfFile(
                mapping,
                windows_sys::Win32::System::Memory::FILE_MAP_READ,
                0,
                0,
                0,
            )
        };

        if ptr.is_null() {
            unsafe {
                windows_sys::Win32::Foundation::CloseHandle(mapping);
            }
            return Err(io::Error::last_os_error());
        }

        Ok(Self {
            ptr: ptr as *mut u8,
            len,
            mapping: mapping as std::os::windows::io::RawHandle,
        })
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

#[cfg(windows)]
impl Drop for MmapInner {
    fn drop(&mut self) {
        unsafe {
            windows_sys::Win32::System::Memory::UnmapViewOfFile(self.ptr as *const _);
            windows_sys::Win32::Foundation::CloseHandle(self.mapping as isize);
        }
    }
}

#[cfg(windows)]
unsafe impl Send for MmapInner {}
#[cfg(windows)]
unsafe impl Sync for MmapInner {}

impl MmapSource {
    /// Memory-map a font file.
    ///
    /// The file is opened in read-only mode and memory-mapped.
    /// Font data is paged in lazily by the OS.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = MmapInner::new(&file)?;
        Ok(Self { mmap })
    }
}

impl FontSource for MmapSource {
    fn as_bytes(&self) -> &[u8] {
        self.mmap.as_slice()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LoadedFont - Owns source and provides Font reference
// ═══════════════════════════════════════════════════════════════════════════

/// A font loaded from a source.
///
/// This struct owns the font source and provides access to a parsed `Font`.
/// The font reference is valid as long as this struct lives.
///
/// # Example
///
/// ```ignore
/// // From embedded data
/// let source = EmbeddedSource::new(include_bytes!("font.ttf"));
/// let loaded = LoadedFont::new(source)?;
/// let glyph = loaded.font().glyph_scaled('A', 16.0);
///
/// // From mmap (large fonts)
/// let source = MmapSource::open("/path/to/large-font.ttf")?;
/// let loaded = LoadedFont::new(source)?;
/// ```
pub struct LoadedFont<S: FontSource> {
    // We store the source to keep the bytes alive.
    // The Font<'a> will borrow from these bytes.
    source: S,
}

impl<S: FontSource> LoadedFont<S> {
    /// Load and parse a font from the given source.
    ///
    /// Returns `None` if the font data is invalid or cannot be parsed.
    pub fn new(source: S) -> Option<Self> {
        // Validate that we can parse the font
        Font::parse(source.as_bytes())?;
        Some(Self { source })
    }

    /// Get a reference to the parsed font.
    ///
    /// The returned font borrows from this `LoadedFont` and is valid
    /// as long as `self` is not dropped.
    pub fn font(&self) -> Font<'_> {
        // SAFETY: We validated the font data in new(), so this should succeed.
        Font::parse(self.source.as_bytes()).expect("font was validated in new()")
    }

    /// Get the raw font data bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.source.as_bytes()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // Use fallback font checked into git (LFS Noto Sans may not be available)
    const FONT_DATA: &[u8] = include_bytes!("../../assets/DejaVuSansMono-Fallback.ttf");

    #[test]
    fn test_embedded_source() {
        let source = EmbeddedSource::new(FONT_DATA);
        assert_eq!(source.as_bytes().len(), FONT_DATA.len());

        let loaded = LoadedFont::new(source).expect("should parse font");
        let font = loaded.font();
        assert!(font.units_per_em > 0);
    }

    #[test]
    fn test_data_source() {
        let source = DataSource::new(FONT_DATA.to_vec());
        assert_eq!(source.as_bytes().len(), FONT_DATA.len());

        let loaded = LoadedFont::new(source).expect("should parse font");
        let font = loaded.font();
        assert!(font.glyph('A').is_some());
    }

    #[test]
    fn test_invalid_font_returns_none() {
        let source = DataSource::new(vec![0, 1, 2, 3]);
        assert!(LoadedFont::new(source).is_none());
    }

    #[test]
    fn test_empty_data_returns_none() {
        let source = DataSource::new(vec![]);
        assert!(LoadedFont::new(source).is_none());
    }

    #[test]
    fn test_glyph_access_through_loaded_font() {
        let source = EmbeddedSource::new(FONT_DATA);
        let loaded = LoadedFont::new(source).expect("should parse font");

        let font = loaded.font();
        let glyph = font.glyph_scaled('A', 16.0);
        assert!(glyph.is_some());

        let advance = font.advance_scaled('A', 16.0);
        assert!(advance.is_some());
    }

    #[cfg(unix)]
    #[test]
    fn test_mmap_source() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a temp file with font data
        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(FONT_DATA).expect("write font data");
        temp.flush().expect("flush");

        let source = MmapSource::open(temp.path()).expect("mmap font file");
        assert_eq!(source.as_bytes().len(), FONT_DATA.len());

        let loaded = LoadedFont::new(source).expect("should parse font");
        let font = loaded.font();
        assert!(font.glyph('A').is_some());
    }
}
