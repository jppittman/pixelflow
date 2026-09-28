// src/io/spawn_macos.rs

//! Spawn-time file actions macOS provides but POSIX does not name.

use std::ffi::{c_char, c_int};

extern "C" {
    // In libSystem since macOS 10.15; the libc crate does not bind it for Apple targets.
    fn posix_spawn_file_actions_addchdir_np(
        actions: *mut libc::posix_spawn_file_actions_t,
        path: *const c_char,
    ) -> c_int;
}

/// Adds a `chdir` to `path` to the child's spawn-time file actions.
///
/// # Safety
/// `actions` must be initialized and `path` a NUL-terminated string that
/// outlives the spawn.
pub(crate) unsafe fn add_chdir(
    actions: *mut libc::posix_spawn_file_actions_t,
    path: *const c_char,
) -> c_int {
    unsafe { posix_spawn_file_actions_addchdir_np(actions, path) }
}
