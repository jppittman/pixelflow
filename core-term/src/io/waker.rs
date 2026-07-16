// src/io/waker.rs

//! Self-pipe [`WakeHandler`] for actors that block in `epoll`/`kqueue` inside
//! `park()`.
//!
//! The actor-scheduler's doorbell is an in-process channel with no file
//! descriptor, so an actor blocked in `epoll_wait`/`kevent` cannot see it.
//! `FdWaker` bridges the gap: the actor registers the waker's read end in its
//! own event monitor, and every `ActorHandle::send()` rings the waker. This is
//! the same pattern the X11/Cocoa display drivers use to interrupt their
//! platform event loops.
//!
//! # Arming protocol
//!
//! A wake byte is only useful while the actor is actually blocked in its
//! poll, so `wake()` skips the pipe-write syscall unless the actor has
//! [`arm`](FdWaker::arm)ed the waker. The owning actor brackets its blocking
//! poll:
//!
//! ```text
//! if !waker.arm() { return Busy }   // wakes raced in: let the scheduler drain
//! monitor.events(&mut events, -1);  // blocks; wake() now writes a byte
//! waker.disarm();                   // stop byte-writes, empty the pipe
//! ... handle events, return Busy ...
//! ```
//!
//! `arm()`/`wake()` use the store-then-check ordering (both sides `SeqCst`) so
//! a wake can never fall between "actor decided to block" and "actor blocked":
//! either the waker sees `polling == true` and writes a byte the poll will
//! see (level-triggered), or `arm()` sees the pending flag and refuses to
//! block. The failure mode this prevents is real — a suppressed wake for a
//! message the previous drain pass left queued would strand it until
//! unrelated fd activity.
//!
//! Draining, conversely, must ONLY happen in `disarm()` (i.e. immediately
//! after the poll returns, with a `Busy` hand-back to the scheduler). Draining
//! anywhere else — e.g. in a message handler — can consume the wake byte of a
//! message that burst limits leave queued, letting the next poll block with
//! nothing left to interrupt it.

use actor_scheduler::WakeHandler;
use anyhow::{Context, Result};
use nix::fcntl::{fcntl, FcntlArg, FdFlag, OFlag};
use std::os::fd::{AsFd, AsRawFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicBool, Ordering};

/// A pipe-based waker: while [`arm`](Self::arm)ed, `wake()` makes the read
/// end readable, unblocking any `epoll_wait`/`kevent` call that includes it
/// in its interest set. While disarmed, `wake()` is a pair of atomic ops and
/// no syscall.
pub struct FdWaker {
    read: OwnedFd,
    write: OwnedFd,
    /// True while the owning actor is blocked (or about to block) in its poll.
    polling: AtomicBool,
    /// Set by every `wake()`; consumed by `arm()` to detect wakes that raced
    /// in while disarmed.
    pending: AtomicBool,
}

impl FdWaker {
    pub fn new() -> Result<Self> {
        let (read, write) = nix::unistd::pipe().context("Failed to create waker pipe")?;
        for fd in [&read, &write] {
            Self::set_nonblocking_cloexec(fd)?;
        }
        Ok(Self {
            read,
            write,
            polling: AtomicBool::new(false),
            pending: AtomicBool::new(false),
        })
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

    /// Declare intent to block in the event monitor.
    ///
    /// Returns `false` if wakes arrived since the last [`disarm`](Self::disarm)
    /// — the caller must skip blocking and return `Busy` so the scheduler
    /// drains its lanes instead.
    #[must_use]
    pub fn arm(&self) -> bool {
        // Order matters (Dekker): publish `polling = true` BEFORE checking
        // `pending`, mirroring wake()'s set-pending-then-check-polling. In the
        // SeqCst total order one side always sees the other.
        self.polling.store(true, Ordering::SeqCst);
        if self.pending.swap(false, Ordering::SeqCst) {
            self.polling.store(false, Ordering::SeqCst);
            return false;
        }
        true
    }

    /// End the blocking poll: stop byte-writes and empty the pipe.
    ///
    /// The caller must return `ActorStatus::Busy` to the scheduler after this
    /// so any message whose wake byte was just drained gets picked up by the
    /// next lane-drain pass.
    pub fn disarm(&self) {
        self.polling.store(false, Ordering::SeqCst);
        self.drain();
    }

    /// Empty the pipe. A wake racing with this may leave a stray byte behind,
    /// which only costs one spurious poll return — never a lost wakeup.
    fn drain(&self) {
        let mut buf = [0u8; 64];
        loop {
            match nix::unistd::read(&self.read, &mut buf) {
                Ok(0) => return,   // write end closed
                Ok(_) => continue, // keep draining
                Err(_) => return,  // EAGAIN: pipe is empty
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
        // Set-pending-then-check-polling; see arm() for the pairing.
        self.pending.store(true, Ordering::SeqCst);
        if !self.polling.load(Ordering::SeqCst) {
            return; // actor isn't in (or entering) a poll; the doorbell suffices
        }
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

    fn pipe_has_byte(waker: &FdWaker) -> bool {
        use std::os::fd::BorrowedFd;
        let mut buf = [0u8; 8];
        // SAFETY: `waker` is borrowed for the duration of this call, so its
        // read fd stays open and valid for the lifetime of `fd`.
        let fd = unsafe { BorrowedFd::borrow_raw(waker.as_raw_fd()) };
        nix::unistd::read(fd, &mut buf).is_ok_and(|n| n > 0)
    }

    #[test]
    fn wake_unblocks_armed_event_monitor() {
        let waker = Arc::new(FdWaker::new().expect("waker"));
        let monitor = EventMonitor::new().expect("monitor");
        monitor
            .add(&*waker, 7, EventFlags::EPOLLIN)
            .expect("register waker");

        assert!(waker.arm(), "no wakes yet, arming should succeed");

        let waker_clone = waker.clone();
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            waker_clone.wake();
        });

        let start = Instant::now();
        let mut events = Vec::new();
        monitor.events(&mut events, 5000).expect("poll");
        waker.disarm();
        handle.join().expect("wake thread");

        assert!(
            start.elapsed() < Duration::from_secs(5),
            "poll should have been interrupted by wake()"
        );
        assert!(
            events.iter().any(|e| e.token == 7),
            "waker token should be reported"
        );
    }

    #[test]
    fn wake_while_disarmed_writes_no_byte_and_fails_next_arm() {
        let waker = FdWaker::new().expect("waker");

        waker.wake();
        assert!(
            !pipe_has_byte(&waker),
            "disarmed wake must not pay the pipe-write syscall"
        );

        // The raced-in wake must surface at arm() instead, so the caller
        // skips blocking and lets the scheduler drain the message.
        assert!(!waker.arm(), "arm must report the missed wake");
        // With the pending flag consumed, arming now succeeds.
        assert!(waker.arm(), "second arm should succeed");
        waker.disarm();
    }

    #[test]
    fn wake_is_idempotent_when_pipe_fills() {
        let waker = FdWaker::new().expect("waker");
        assert!(waker.arm());
        // Far more wakes than the pipe buffer holds; must not block or panic.
        for _ in 0..100_000 {
            waker.wake();
        }
        waker.disarm(); // also drains
        assert!(!pipe_has_byte(&waker), "disarm should empty the pipe");
    }
}
