// src/io/event_monitor_actor/reader.rs

//! PTY reader actor.
//!
//! Owns a clone of the PTY master FD and a fixed pool of read buffers. Its
//! **data lane is the buffer recycler**: the parser returns drained buffers as
//! `Data` messages, and the pool is bootstrapped the same way at spawn.
//!
//! `park()` is the bridge to the OS: it blocks in `epoll_wait`/`kevent` on
//! `{pty, waker}` and returns `Busy` so the scheduler polls the doorbell with
//! `try_recv` instead of blocking on it. The two idle cases return `Idle` and
//! let the scheduler block on the doorbell instead:
//!
//! - **Pool empty**: reading is impossible; only a recycled buffer (a doorbell
//!   event) can make progress. Unread PTY output waits in the kernel buffer,
//!   which blocks the shell — that is the backpressure chain working.
//! - **EOF seen**: the PTY is deregistered; only `Shutdown` is left to handle.
//!
//! On EOF (or a fatal read error) the reader notifies the app via
//! [`PtySender::send_child_exited`] so the terminal can quit instead of
//! sitting on a dead session.

use super::{FilledBuf, NoControl, NoManagement, ParserControl};
use crate::io::event::{EventFlags, EventMonitor};
use crate::io::pty::NixPty;
use crate::io::traits::PtySender;
use crate::io::waker::FdWaker;
use actor_scheduler::{
    Actor, ActorHandle, ActorStatus, HandlerError, HandlerResult, Message, SystemStatus,
};
use log::*;
use std::io::Read;
use std::sync::Arc;

/// Size of each read buffer. One kernel PTY buffer's worth is plenty.
pub(super) const READ_BUFFER_SIZE: usize = 4096;

/// Number of buffers in circulation. Bounds both memory and the number of
/// unparsed batches in flight; the reader stops reading when all are out.
pub(super) const POOL_SIZE: usize = 8;

const TOKEN_PTY: u64 = 0;
const TOKEN_WAKER: u64 = 1;

pub(super) struct PtyReader {
    pty: NixPty,
    monitor: EventMonitor,
    waker: Arc<FdWaker>,
    /// Buffers available for reading. Refilled via the data lane.
    /// Invariant: every buffer has `len() == READ_BUFFER_SIZE` — buffers are
    /// never resized, so no read ever pays a re-zeroing memset.
    pool: Vec<Vec<u8>>,
    parser_tx: ActorHandle<FilledBuf, ParserControl, NoManagement>,
    app_tx: Box<dyn PtySender>,
    events: Vec<crate::io::event::Event>,
    eof: bool,
}

impl PtyReader {
    /// Builds the reader and registers the PTY and waker with the event
    /// monitor. Called on the reader thread (the monitor stays thread-local).
    pub(super) fn new(
        pty: NixPty,
        waker: Arc<FdWaker>,
        parser_tx: ActorHandle<FilledBuf, ParserControl, NoManagement>,
        app_tx: Box<dyn PtySender>,
    ) -> anyhow::Result<Self> {
        let monitor = EventMonitor::new()?;
        monitor.add(&pty, TOKEN_PTY, EventFlags::EPOLLIN)?;
        monitor.add(&*waker, TOKEN_WAKER, EventFlags::EPOLLIN)?;
        Ok(Self {
            pty,
            monitor,
            waker,
            pool: Vec::with_capacity(POOL_SIZE),
            parser_tx,
            app_tx,
            events: Vec::with_capacity(16),
            eof: false,
        })
    }

    /// Read until the PTY would block or the pool runs dry.
    fn read_available(&mut self) {
        while let Some(mut buf) = self.pool.pop() {
            match self.pty.read(&mut buf) {
                Ok(0) => {
                    self.pool.push(buf);
                    info!("PTY returned EOF, child exited");
                    self.mark_eof();
                    return;
                }
                Ok(len) => {
                    if let Err(e) = self.parser_tx.send(Message::Data(FilledBuf { data: buf, len }))
                    {
                        warn!("PTY reader: parser channel closed: {}", e);
                        self.mark_eof();
                        return;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    self.pool.push(buf);
                    return;
                }
                Err(e) => {
                    self.pool.push(buf);
                    error!("PTY read error: {}", e);
                    self.mark_eof();
                    return;
                }
            }
        }
    }

    fn mark_eof(&mut self) {
        self.eof = true;
        if let Err(e) = self.monitor.delete(&self.pty) {
            debug!("PTY reader: failed to deregister PTY fd: {}", e);
        }
        if let Err(e) = self.app_tx.send_child_exited() {
            warn!("PTY reader: failed to notify app of child exit: {}", e);
        }
    }
}

impl Actor<Vec<u8>, NoControl, NoManagement> for PtyReader {
    fn handle_data(&mut self, mut buf: Vec<u8>) -> HandlerResult {
        if self.eof {
            return Ok(()); // pool is no longer needed; let buffers drop
        }
        // Restore the pool invariant if a producer hands us a nonconforming
        // buffer; the cost lands here (once per recycle, normally a no-op)
        // instead of on every read.
        if buf.len() != READ_BUFFER_SIZE {
            buf.resize(READ_BUFFER_SIZE, 0);
        }
        self.pool.push(buf);
        Ok(())
    }

    fn handle_control(&mut self, _msg: NoControl) -> HandlerResult {
        Ok(())
    }

    fn handle_management(&mut self, _msg: NoManagement) -> HandlerResult {
        Ok(())
    }

    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        if self.eof || self.pool.is_empty() {
            return Ok(ActorStatus::Idle);
        }

        // Wakes that raced in while we were processing mean messages are
        // queued: hand back Busy so the scheduler drains them now instead of
        // blocking with their wake bytes already consumed.
        if !self.waker.arm() {
            return Ok(ActorStatus::Busy);
        }

        let poll = self.monitor.events(&mut self.events, -1);
        self.waker.disarm();
        poll.map_err(|e| HandlerError::recoverable(format!("PTY reader poll failed: {e}")))?;

        let mut pty_ready = false;
        for event in &self.events {
            match event.token {
                TOKEN_WAKER => {} // already drained by disarm()
                TOKEN_PTY => pty_ready = true,
                other => debug!("PTY reader: unexpected poll token {}", other),
            }
        }

        if pty_ready {
            self.read_available();
        }

        // Busy: this actor's real doorbell is the event monitor, so the
        // scheduler must come back through park() instead of blocking on
        // the channel doorbell.
        Ok(ActorStatus::Busy)
    }
}
