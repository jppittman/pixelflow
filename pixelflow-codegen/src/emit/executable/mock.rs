//! Mock code page implementation for testing without OS page permissions.

use super::{CodePage, ExecutableCode};
use crate::error::CompileError;
use alloc::vec::Vec;

/// An in-memory code page for testing.
#[derive(Clone, Debug)]
pub struct MockCodePage {
    pub bytes: Vec<u8>,
    pub capacity: usize,
}

impl CodePage for MockCodePage {
    fn page_size() -> usize {
        4096
    }

    fn map(capacity: usize) -> Result<Self, CompileError> {
        Ok(Self {
            bytes: Vec::with_capacity(capacity),
            capacity,
        })
    }

    fn write(&mut self, code: &[u8]) {
        self.bytes.extend_from_slice(code);
    }

    fn finish(self, len: usize) -> Result<ExecutableCode, CompileError> {
        use libc::{MAP_ANON, MAP_PRIVATE, PROT_READ, PROT_WRITE, mmap};

        let capacity = (len.max(1) + 4095) & !4095;
        let ptr = unsafe {
            mmap(
                core::ptr::null_mut(),
                capacity,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(CompileError::Mmap);
        }
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.bytes.as_ptr(),
                ptr.cast(),
                len.min(self.bytes.len()),
            )
        };
        Ok(ExecutableCode {
            ptr: ptr.cast(),
            len,
            capacity,
        })
    }
}
