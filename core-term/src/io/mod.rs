//! # PTY I/O
//!
//! Pseudo-terminal plumbing: PTY creation ([`pty`]), platform event
//! notification ([`event`]: epoll on Linux, kqueue on macOS), the self-pipe
//! waker that bridges the actor scheduler to those polls ([`waker`]), and the
//! three-actor PTY pipeline ([`event_monitor_actor`]).
//!
//! The pipeline architecture — reader, parser, and writer actors, their
//! lanes, backpressure, and lifecycle — is documented on
//! [`event_monitor_actor`].

pub mod event_monitor_actor;
pub mod pty;
pub mod traits;
pub mod waker;

/// Represents dimensions for a resize command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resize {
    /// Number of columns in the terminal grid.
    pub cols: u16,
    /// Number of rows in the terminal grid.
    pub rows: u16,
}

#[cfg(test)]
mod pty_tests;

// Platform-specific event monitoring implementations
#[cfg(target_os = "macos")]
pub mod kqueue;

#[cfg(target_os = "linux")]
pub mod epoll;

// Platform-agnostic re-exports. `Event`/`EventFlags` alias the platform types
// so poll loops can be written once; the flag names match epoll on both.
#[cfg(target_os = "macos")]
pub mod event {
    pub use super::kqueue::*;
    pub type Event = KqueueEvent;
    pub type EventFlags = KqueueFlags;
}

#[cfg(target_os = "linux")]
pub mod event {
    pub use super::epoll::*;
    pub type Event = EpollEvent;
    pub type EventFlags = EpollFlags;
}
