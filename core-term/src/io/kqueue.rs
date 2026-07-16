// src/platform/os/kqueue.rs

//! This module provides a wrapper around `kqueue` functionality for macOS,
//! providing the same interface as the epoll module for Linux.
//! It defines type-safe enums and bitflags for kqueue operations and events.

use anyhow::{Context, Result};
use bitflags::bitflags;
use log::{debug, trace};
use std::io;
use std::os::unix::io::RawFd;

bitflags! {
    /// Event flags matching the semantics of EpollFlags for API compatibility
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct KqueueFlags: u16 {
        const EPOLLIN = 0x0001;   // Monitor for read readiness (matches epoll naming)
        const EPOLLOUT = 0x0002;  // Monitor for write readiness (matches epoll naming)
    }
}

/// kqueue event structure that we'll use
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KqueueEvent {
    pub token: u64,
    pub flags: KqueueFlags,
}

/// EventMonitor for macOS using kqueue
#[derive(Debug)]
pub struct EventMonitor {
    kqueue_fd: RawFd,
}

impl EventMonitor {
    pub fn new() -> Result<Self> {
        let kqueue_fd = unsafe { libc::kqueue() };
        if kqueue_fd == -1 {
            return Err(io::Error::last_os_error()).context("Failed to create kqueue instance");
        }
        debug!("EventMonitor created with kqueue_fd: {}", kqueue_fd);
        Ok(Self { kqueue_fd })
    }

    pub fn add<S: std::os::unix::io::AsRawFd>(
        &self,
        source: &S,
        token: u64,
        flags: KqueueFlags,
    ) -> Result<()> {
        let fd = source.as_raw_fd();
        let mut changes = Vec::new();

        // Add read filter if requested
        if flags.contains(KqueueFlags::EPOLLIN) {
            let mut kev: libc::kevent = unsafe { std::mem::zeroed() };
            kev.ident = fd as usize;
            kev.filter = libc::EVFILT_READ;
            kev.flags = libc::EV_ADD | libc::EV_ENABLE;
            kev.udata = token as *mut libc::c_void;
            changes.push(kev);
        }

        // Add write filter if requested
        if flags.contains(KqueueFlags::EPOLLOUT) {
            let mut kev: libc::kevent = unsafe { std::mem::zeroed() };
            kev.ident = fd as usize;
            kev.filter = libc::EVFILT_WRITE;
            kev.flags = libc::EV_ADD | libc::EV_ENABLE;
            kev.udata = token as *mut libc::c_void;
            changes.push(kev);
        }

        if !changes.is_empty() {
            let ret = unsafe {
                libc::kevent(
                    self.kqueue_fd,
                    changes.as_ptr(),
                    changes.len() as i32,
                    std::ptr::null_mut(),
                    0,
                    std::ptr::null(),
                )
            };
            if ret == -1 {
                return Err(io::Error::last_os_error()).with_context(|| {
                    format!("Failed to add fd {} to kqueue (token: {})", fd, token)
                });
            }
        }

        trace!(
            "Added fd {} to kqueue_fd {} with token {} and flags {:?}",
            fd,
            self.kqueue_fd,
            token,
            flags
        );
        Ok(())
    }

    pub fn modify<S: std::os::unix::io::AsRawFd>(
        &self,
        source: &S,
        token: u64,
        flags: KqueueFlags,
    ) -> Result<()> {
        // For kqueue, modify is essentially delete + add
        self.delete(source)?;
        self.add(source, token, flags)
    }

    pub fn delete<S: std::os::unix::io::AsRawFd>(&self, source: &S) -> Result<()> {
        let fd = source.as_raw_fd();
        let mut changes = Vec::new();

        // Delete both read and write filters
        for filter in [libc::EVFILT_READ, libc::EVFILT_WRITE] {
            let mut kev: libc::kevent = unsafe { std::mem::zeroed() };
            kev.ident = fd as usize;
            kev.filter = filter;
            kev.flags = libc::EV_DELETE;
            changes.push(kev);
        }

        let ret = unsafe {
            libc::kevent(
                self.kqueue_fd,
                changes.as_ptr(),
                changes.len() as i32,
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
            )
        };

        // It's okay if delete fails (fd might not have been registered for that filter)
        if ret == -1 {
            let err = io::Error::last_os_error();
            // ENOENT means the event wasn't registered, which is fine
            if err.raw_os_error() != Some(libc::ENOENT) {
                return Err(err).with_context(|| format!("Failed to delete fd {} from kqueue", fd));
            }
        }

        trace!("Deleted fd {} from kqueue_fd {}", fd, self.kqueue_fd);
        Ok(())
    }

    pub fn events(&self, events_out: &mut Vec<KqueueEvent>, timeout_ms: isize) -> Result<()> {
        trace!(
            "EventMonitor: polling for events with timeout {}ms on kqueue_fd {}",
            timeout_ms,
            self.kqueue_fd
        );

        // Prepare storage for kernel events
        const MAX_EVENTS: usize = 32;
        let mut kevents: [libc::kevent; MAX_EVENTS] = unsafe { std::mem::zeroed() };

        // RAII: Keep timespec alive for entire scope
        // Using Option to represent "no timeout" (None) vs "timeout" (Some)
        let timeout_spec = if timeout_ms < 0 {
            None
        } else {
            Some(libc::timespec {
                tv_sec: (timeout_ms / 1000) as libc::time_t,
                tv_nsec: ((timeout_ms % 1000) * 1_000_000) as libc::c_long,
            })
        };

        // Safe: timeout_spec owns the data, reference lives as long as timeout_spec
        let timeout = timeout_spec
            .as_ref()
            .map(|ts| ts as *const libc::timespec)
            .unwrap_or(std::ptr::null());

        let nev = unsafe {
            libc::kevent(
                self.kqueue_fd,
                std::ptr::null(),
                0,
                kevents.as_mut_ptr(),
                MAX_EVENTS as i32,
                timeout,
            )
        };

        if nev == -1 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted {
                // EINTR is not a fatal error, just return empty events
                events_out.clear();
                return Ok(());
            }
            return Err(err).context("kevent() call failed");
        }

        trace!("kevent returned {} events", nev);

        // Convert kernel events to our events
        events_out.clear();

        for i in 0..nev as usize {
            let kev = &kevents[i];
            let token = kev.udata as u64;

            let mut flags = KqueueFlags::empty();
            if kev.filter == libc::EVFILT_READ {
                flags |= KqueueFlags::EPOLLIN;
            }
            if kev.filter == libc::EVFILT_WRITE {
                flags |= KqueueFlags::EPOLLOUT;
            }

            // Optimization: avoid HashSet allocation for small N (MAX_EVENTS=32)
            if let Some(event) = events_out.iter_mut().find(|e| e.token == token) {
                event.flags |= flags;
            } else {
                events_out.push(KqueueEvent { token, flags });
            }
        }

        Ok(())
    }
}

impl Drop for EventMonitor {
    fn drop(&mut self) {
        trace!("Closing kqueue_fd {}", self.kqueue_fd);
        unsafe {
            libc::close(self.kqueue_fd);
        }
    }
}
