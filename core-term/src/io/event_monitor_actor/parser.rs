// src/io/event_monitor_actor/parser.rs

//! PTY parser actor.
//!
//! Receives raw byte batches from the reader on its data lane, parses them
//! into an `AnsiBatch` (the CPU-heavy step), forwards the batch to the app,
//! and recycles the drained buffer back to the reader's data lane. Purely
//! message-driven — no OS blocking, so no waker.
//!
//! The reader-recycle handle comes from the troupe `Directory`; the app sink
//! arrives via the `Bind` management message.

use super::{Directory, FilledBuf, ParserControl, ParserManagement, ReaderManagement};
use crate::ansi::{AnsiParser, AnsiProcessor};
use crate::io::traits::PtySender;
use actor_scheduler::{
    Actor, ActorHandle, ActorStatus, ActorTypes, HandlerError, HandlerResult, Message,
    SystemStatus, TroupeActor,
};
use log::*;
use std::convert::Infallible;

/// Handle to the reader for buffer recycling (from the troupe `Directory`).
type ReaderTx = ActorHandle<Vec<u8>, Infallible, ReaderManagement>;

pub(super) struct PtyParser {
    parser: AnsiProcessor,
    reader_tx: ReaderTx,
    /// App sink, delivered by `Bind`.
    app_tx: Option<Box<dyn PtySender>>,
}

impl ActorTypes for PtyParser {
    type Data = FilledBuf;
    type Control = ParserControl;
    type Management = ParserManagement;
}

impl TroupeActor<Directory> for PtyParser {
    fn new(dir: Directory) -> Self {
        Self {
            parser: AnsiProcessor::new(),
            reader_tx: dir.reader,
            app_tx: None,
        }
    }
}

impl Actor<FilledBuf, ParserControl, ParserManagement> for PtyParser {
    fn handle_data(&mut self, filled: FilledBuf) -> HandlerResult {
        let parsed = self.parser.process_bytes(filled.bytes());

        // Recycle the buffer; if the reader is gone the buffer just drops.
        if let Err(e) = self.reader_tx.send(Message::Data(filled.data)) {
            debug!("PTY parser: reader gone, dropping recycled buffer: {}", e);
        }

        if parsed.is_empty() {
            return Ok(());
        }
        match &self.app_tx {
            // Bind (Management) always drains before Data, so this is set.
            Some(app_tx) => {
                if let Err(e) = app_tx.send(parsed) {
                    warn!("PTY parser: failed to send commands to app: {}", e);
                }
            }
            None => warn!("PTY parser: received data before Bind; dropping commands"),
        }
        Ok(())
    }

    fn handle_control(&mut self, msg: ParserControl) -> HandlerResult {
        match msg {
            ParserControl::Reset => {
                // Drop any half-parsed escape sequence state.
                self.parser = AnsiProcessor::new();
            }
        }
        Ok(())
    }

    fn handle_management(&mut self, msg: ParserManagement) -> HandlerResult {
        let ParserManagement::Bind { app_tx } = msg;
        self.app_tx = Some(app_tx);
        Ok(())
    }

    fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        // Purely message-driven; the scheduler may block on the doorbell.
        Ok(ActorStatus::Idle)
    }
}
