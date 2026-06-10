use crate::api::private::{EngineActorHandle, EngineControl, EngineData};
use crate::api::public::AppManagement; // Use public re-export
use actor_scheduler::{
    Actor, ActorBuilder, ActorStatus, HandlerError, HandlerResult, SystemStatus,
};
use std::sync::{Arc, Mutex};

/// A recorded message received by the MockEngine.
#[derive(Debug)]
pub enum ReceivedMessage {
    Data(EngineData),
    Control(EngineControl),
    Management(AppManagement),
}

/// A mock engine that captures messages sent to it suitable for unit testing actors.
pub struct MockEngine {
    messages: Arc<Mutex<Vec<ReceivedMessage>>>,
    handle: Option<EngineActorHandle>,
    _thread: Option<std::thread::JoinHandle<()>>,
}

impl Default for MockEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl MockEngine {
    /// Create a new MockEngine. Returns the engine instance (to inspect messages)
    /// and the handle (to pass to the actor under test).
    #[must_use]
    pub fn new() -> Self {
        let messages = Arc::new(Mutex::new(Vec::new()));

        let mut collector = MessageCollector {
            messages: messages.clone(),
        };

        // Create scheduler with two producers: one for self, one for handle()
        let mut builder = ActorBuilder::<EngineData, EngineControl, AppManagement>::new(100, None);
        let _handle = builder.add_producer();
        let extra_handle = builder.add_producer();
        let mut scheduler = builder.build();

        // Spawn background thread to process messages
        let thread = std::thread::spawn(move || {
            scheduler.run(&mut collector);
        });

        Self {
            messages,
            handle: Some(extra_handle),
            _thread: Some(thread),
        }
    }

    /// Take the dedicated SPSC handle to the mock engine.
    ///
    /// With SPSC channels, each handle is a unique producer.
    /// This can only be called once — panics if called again.
    pub fn take_handle(&mut self) -> EngineActorHandle {
        self.handle.take().expect("MockEngine handle already taken")
    }

    pub fn messages(&self) -> std::sync::MutexGuard<'_, Vec<ReceivedMessage>> {
        self.messages.lock().unwrap()
    }
}

// Collector Actor
struct MessageCollector {
    messages: Arc<Mutex<Vec<ReceivedMessage>>>,
}

impl Actor<EngineData, EngineControl, AppManagement> for MessageCollector {
    fn handle_data(&mut self, msg: EngineData) -> HandlerResult {
        self.messages
            .lock()
            .unwrap()
            .push(ReceivedMessage::Data(msg));
        Ok(())
    }

    fn handle_control(&mut self, msg: EngineControl) -> HandlerResult {
        self.messages
            .lock()
            .unwrap()
            .push(ReceivedMessage::Control(msg));
        Ok(())
    }

    fn handle_management(&mut self, msg: AppManagement) -> HandlerResult {
        self.messages
            .lock()
            .unwrap()
            .push(ReceivedMessage::Management(msg));
        Ok(())
    }

    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}
