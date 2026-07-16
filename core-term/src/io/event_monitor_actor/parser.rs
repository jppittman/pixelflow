// src/io/event_monitor_actor/parser.rs

//! PTY parser actor.
//!
//! Receives raw byte batches from the reader on its data lane, parses them
//! into `AnsiCommand`s (the CPU-heavy step), forwards the commands to the app,
//! and recycles the drained buffer back to the reader's data lane.

use super::{NoControl, NoManagement, ParserControl};
use crate::ansi::{AnsiParser, AnsiProcessor};
use crate::io::traits::PtySender;
use actor_scheduler::{
    Actor, ActorHandle, ActorStatus, HandlerError, HandlerResult, Message, SystemStatus,
};
use log::*;

pub(super) struct PtyParser {
    parser: AnsiProcessor,
    app_tx: Box<dyn PtySender>,
    /// Recycler: drained buffers go back to the reader's pool.
    reader_tx: ActorHandle<Vec<u8>, NoControl, NoManagement>,
}

impl PtyParser {
    pub(super) fn new(
        app_tx: Box<dyn PtySender>,
        reader_tx: ActorHandle<Vec<u8>, NoControl, NoManagement>,
    ) -> Self {
        Self {
            parser: AnsiProcessor::new(),
            app_tx,
            reader_tx,
        }
    }
}

impl Actor<Vec<u8>, ParserControl, NoManagement> for PtyParser {
    fn handle_data(&mut self, data: Vec<u8>) -> HandlerResult {
        let commands = self.parser.process_bytes(&data);

        // Recycle the buffer; if the reader is gone the buffer just drops.
        if let Err(e) = self.reader_tx.send(Message::Data(data)) {
            debug!("PTY parser: reader gone, dropping recycled buffer: {}", e);
        }

        if !commands.is_empty() {
            if let Err(e) = self.app_tx.send(commands) {
                warn!("PTY parser: failed to send commands to app: {}", e);
            }
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

    fn handle_management(&mut self, _msg: NoManagement) -> HandlerResult {
        Ok(())
    }

    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        // Purely message-driven; the scheduler may block on the doorbell.
        Ok(ActorStatus::Idle)
    }
}
