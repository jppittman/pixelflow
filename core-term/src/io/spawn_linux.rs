// src/io/spawn_linux.rs

//! Spawn-time file actions glibc provides but POSIX does not name.

use std::ffi::{c_char, c_int};

/// Adds a `chdir` to `path` to the child's spawn-time file actions.
///
/// # Safety
/// `actions` must be initialized and `path` a NUL-terminated string that
/// outlives the spawn.
pub(crate) unsafe fn add_chdir(
    actions: *mut libc::posix_spawn_file_actions_t,
    path: *const c_char,
) -> c_int {
    unsafe { libc::posix_spawn_file_actions_addchdir_np(actions, path) }
}
