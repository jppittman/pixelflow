// src/io/waker.rs

//! Self-pipe [`WakeHandler`] for actors that block in `epoll`/`kqueue` inside
//! `park()`.
//!
//! The actor-scheduler's doorbell is an in-process channel with no file
//! descriptor, so an actor blocked in `epoll_wait`/`kevent` cannot see it.
//! `FdWaker` bridges the gap: the actor registers the waker's read end in its
//! own event monitor, and every `ActorHandle::send()` rings the waker by
//! writing a byte to the write end. This is the same pattern the X11/Cocoa
//! display drivers use to interrupt their platform event loops.

use actor_scheduler::WakeHandler;
use anyhow::{Context, Result};
use nix::fcntl::{fcntl, FcntlArg, FdFlag, OFlag};
use std::os::fd::{AsFd, AsRawFd, OwnedFd, RawFd};

/// A pipe-based waker: `wake()` makes the read end readable, unblocking any
/// `epoll_wait`/`kevent` call that includes it in its interest set.
pub struct FdWaker {
    read: OwnedFd,
    write: OwnedFd,
}

impl FdWaker {
    pub fn new() -> Result<Self> {
        let (read, write) = nix::unistd::pipe().context("Failed to create waker pipe")?;
        for fd in [&read, &write] {
            Self::set_nonblocking_cloexec(fd)?;
        }
        Ok(Self { read, write })
    }

    fn set_nonblocking_cloexec(fd: &OwnedFd) -> Result<()> {
        let raw_fd = fd.as_raw_fd();
        fcntl(fd.as_fd(), FcntlArg::F_SETFD(FdFlag::FD_CLOEXEC))
            .with_context(|| format!("Failed to set FD_CLOEXEC on waker fd {}", raw_fd))?;
        let flags = fcntl(fd.as_fd(), FcntlArg::F_GETFL)
            .with_context(|| format!("Failed to get flags for waker fd {}", raw_fd))?;
        let mut flags = OFlag::from_bits_truncate(flags);
        flags.insert(OFlag::O_NONBLOCK);
        fcntl(fd.as_fd(), FcntlArg::F_SETFL(flags))
            .with_context(|| format!("Failed to set O_NONBLOCK on waker fd {}", raw_fd))?;
        Ok(())
    }

    /// Empty the pipe after a wakeup so the next `wake()` triggers a fresh edge.
    pub fn drain(&self) {
        let mut buf = [0u8; 64];
        loop {
            match nix::unistd::read(&self.read, &mut buf) {
                Ok(0) => return,          // write end closed
                Ok(_) => continue,        // keep draining
                Err(_) => return,         // EAGAIN: pipe is empty
            }
        }
    }
}

/// The read end, for registration in an `EventMonitor`.
impl AsRawFd for FdWaker {
    fn as_raw_fd(&self) -> RawFd {
        self.read.as_raw_fd()
    }
}

impl WakeHandler for FdWaker {
    fn wake(&self) {
        match nix::unistd::write(&self.write, &[1u8]) {
            Ok(_) => {}
            // Pipe full: a wake is already pending, which is all we need.
            Err(nix::Error::EAGAIN) => {}
            Err(e) => log::warn!("FdWaker: failed to write wake byte: {}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::event::EventMonitor;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[cfg(target_os = "linux")]
    use crate::io::event::EpollFlags as EventFlags;
    #[cfg(target_os = "macos")]
    use crate::io::event::KqueueFlags as EventFlags;

    #[test]
    fn wake_unblocks_event_monitor() {
        let waker = Arc::new(FdWaker::new().expect("waker"));
        let monitor = EventMonitor::new().expect("monitor");
        monitor
            .add(&*waker, 7, EventFlags::EPOLLIN)
            .expect("register waker");

        let waker_clone = waker.clone();
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            waker_clone.wake();
        });

        let start = Instant::now();
        let mut events = Vec::new();
        monitor.events(&mut events, 5000).expect("poll");
        handle.join().expect("wake thread");

        assert!(
            start.elapsed() < Duration::from_secs(5),
            "poll should have been interrupted by wake()"
        );
        assert!(
            events.iter().any(|e| e.token == 7),
            "waker token should be reported"
        );
        waker.drain();
    }

    #[test]
    fn wake_is_idempotent_when_pipe_fills() {
        let waker = FdWaker::new().expect("waker");
        // Far more wakes than the pipe buffer holds; must not block or panic.
        for _ in 0..100_000 {
            waker.wake();
        }
        waker.drain();
        // After drain, another wake still lands.
        waker.wake();
        let mut buf = [0u8; 8];
        let n = nix::unistd::read(&waker.read, &mut buf).expect("read");
        assert!(n > 0, "wake after drain should write a byte");
    }
}
