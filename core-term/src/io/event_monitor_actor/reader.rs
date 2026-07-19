// src/io/event_monitor_actor/reader.rs

//! PTY reader actor.
//!
//! Owns a clone of the PTY master FD and a fixed pool of read buffers. Its
//! **data lane is the buffer recycler**: the parser returns drained buffers as
//! `Data` messages. The pool is allocated when the `Bind` management message
//! arrives (see [`super`] for the bind protocol).
//!
//! `park()` is the bridge to the OS: it blocks in `epoll_wait`/`kevent` on
//! `{pty, waker}` and returns `Busy` so the scheduler polls the doorbell with
//! `try_recv` instead of blocking on it. Three cases return `Idle` and let the
//! scheduler block on the doorbell instead:
//!
//! - **Unbound**: no PTY yet; only the `Bind` message can make progress.
//! - **Pool empty**: reading is impossible; only a recycled buffer (a doorbell
//!   event) can. Unread PTY output waits in the kernel buffer, which blocks the
//!   shell — that is the backpressure chain working.
//! - **EOF seen**: the PTY is deregistered; only `Shutdown` is left to handle.
//!
//! On EOF (or a fatal read error) the reader notifies the app via
//! [`PtySender::send_child_exited`] so the terminal can quit instead of
//! sitting on a dead session.

use super::{Directory, FilledBuf, NoControl, ParserControl, ParserManagement, ReaderManagement};
use crate::io::event::{Event, EventFlags, EventMonitor};
use crate::io::pty::NixPty;
use crate::io::traits::PtySender;
use crate::io::waker::FdWaker;
use actor_scheduler::{
    Actor, ActorHandle, ActorStatus, ActorTypes, HandlerError, HandlerResult, Message,
    SystemStatus, TroupeActor,
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

/// Handle to the parser (from the troupe `Directory`).
type ParserTx = ActorHandle<FilledBuf, ParserControl, ParserManagement>;

pub(super) struct PtyReader {
    parser_tx: ParserTx,
    bound: Option<BoundReader>,
}

/// Runtime resources, present once the `Bind` message has been handled.
struct BoundReader {
    pty: NixPty,
    monitor: EventMonitor,
    waker: Arc<FdWaker>,
    app_tx: Box<dyn PtySender>,
    /// Buffers available for reading.
    /// Invariant: every buffer has `len() == READ_BUFFER_SIZE` — buffers are
    /// never resized, so no read ever pays a re-zeroing memset.
    pool: Vec<Vec<u8>>,
    events: Vec<Event>,
    eof: bool,
}

impl BoundReader {
    fn bind(pty: NixPty, waker: Arc<FdWaker>, app_tx: Box<dyn PtySender>) -> anyhow::Result<Self> {
        let monitor = EventMonitor::new()?;
        monitor.add(&pty, TOKEN_PTY, EventFlags::EPOLLIN)?;
        monitor.add(&*waker, TOKEN_WAKER, EventFlags::EPOLLIN)?;
        // Seed the pool. These are the only buffer allocations the pipeline
        // ever makes; they ship at full length so reads never resize them.
        let pool = (0..POOL_SIZE)
            .map(|_| vec![0u8; READ_BUFFER_SIZE])
            .collect();
        Ok(Self {
            pty,
            monitor,
            waker,
            app_tx,
            pool,
            events: Vec::with_capacity(16),
            eof: false,
        })
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

impl PtyReader {
    /// Read until the PTY would block or the pool runs dry, forwarding each
    /// batch to the parser.
    fn read_available(&mut self) {
        let Some(bound) = self.bound.as_mut() else {
            return;
        };
        while let Some(mut buf) = bound.pool.pop() {
            match bound.pty.read(&mut buf) {
                Ok(0) => {
                    bound.pool.push(buf);
                    info!("PTY returned EOF, child exited");
                    bound.mark_eof();
                    return;
                }
                Ok(len) => {
                    // `bound` borrows self.bound; self.parser_tx is a disjoint
                    // field, so this send is allowed alongside the borrow.
                    if let Err(e) = self
                        .parser_tx
                        .send(Message::Data(FilledBuf { data: buf, len }))
                    {
                        warn!("PTY reader: parser channel closed: {}", e);
                        bound.mark_eof();
                        return;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    bound.pool.push(buf);
                    return;
                }
                Err(e) => {
                    bound.pool.push(buf);
                    error!("PTY read error: {}", e);
                    bound.mark_eof();
                    return;
                }
            }
        }
    }
}

impl ActorTypes for PtyReader {
    type Data = Vec<u8>;
    type Control = NoControl;
    type Management = ReaderManagement;
}

impl TroupeActor<Directory> for PtyReader {
    fn new(dir: Directory) -> Self {
        Self {
            parser_tx: dir.parser,
            bound: None,
        }
    }
}

impl Actor<Vec<u8>, NoControl, ReaderManagement> for PtyReader {
    fn handle_data(&mut self, mut buf: Vec<u8>) -> HandlerResult {
        let Some(bound) = self.bound.as_mut() else {
            return Ok(()); // not bound yet; drop (no reader data arrives pre-bind)
        };
        if bound.eof {
            return Ok(()); // pool no longer needed; let recycled buffers drop
        }
        // Restore the pool invariant if a producer hands us a nonconforming
        // buffer; the cost lands here (once per recycle, normally a no-op)
        // instead of on every read.
        if buf.len() != READ_BUFFER_SIZE {
            buf.resize(READ_BUFFER_SIZE, 0);
        }
        bound.pool.push(buf);
        Ok(())
    }

    fn handle_control(&mut self, _msg: NoControl) -> HandlerResult {
        Ok(())
    }

    fn handle_management(&mut self, msg: ReaderManagement) -> HandlerResult {
        let ReaderManagement::Bind { pty, waker, app_tx } = msg;
        match BoundReader::bind(pty, waker, app_tx) {
            Ok(bound) => self.bound = Some(bound),
            Err(e) => {
                return Err(HandlerError::recoverable(format!(
                    "PTY reader bind failed: {e}"
                )))
            }
        }
        Ok(())
    }

    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        let Some(bound) = self.bound.as_mut() else {
            return Ok(ActorStatus::Idle); // unbound: wait for Bind on the doorbell
        };
        if bound.eof || bound.pool.is_empty() {
            return Ok(ActorStatus::Idle);
        }

        // Wakes that raced in while we were processing mean messages are
        // queued: hand back Busy so the scheduler drains them now instead of
        // blocking with their wake bytes already consumed.
        if !bound.waker.arm() {
            return Ok(ActorStatus::Busy);
        }

        let poll = bound.monitor.events(&mut bound.events, -1);
        bound.waker.disarm();
        poll.map_err(|e| HandlerError::recoverable(format!("PTY reader poll failed: {e}")))?;

        let mut pty_ready = false;
        for event in &bound.events {
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
