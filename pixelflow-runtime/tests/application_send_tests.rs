//! `Application::send` is how the engine reports events to the app: each
//! `EngineEvent` must arrive on the lane its variant names.

use actor_scheduler::{
    Actor, ActorScheduler, ActorStatus, HandlerError, HandlerResult, SystemStatus,
};
use pixelflow_runtime::api::public::{
    Application, EngineEvent, EngineEventControl, EngineEventData, EngineEventManagement,
};
use pixelflow_runtime::display::messages::WindowId;
use pixelflow_runtime::input::{KeySymbol, Modifiers};
use std::time::{Duration, Instant};

/// Records which lane each delivered event arrived on.
#[derive(Default)]
struct LaneLog {
    lanes: Vec<&'static str>,
}

impl Actor<EngineEventData, EngineEventControl, EngineEventManagement> for LaneLog {
    fn handle_data(&mut self, _msg: EngineEventData) -> HandlerResult {
        self.lanes.push("data");
        Ok(())
    }

    fn handle_control(&mut self, _msg: EngineEventControl) -> HandlerResult {
        self.lanes.push("control");
        Ok(())
    }

    fn handle_management(&mut self, _msg: EngineEventManagement) -> HandlerResult {
        self.lanes.push("management");
        Ok(())
    }

    fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

#[test]
fn application_send_delivers_each_engine_event_on_the_lane_of_its_variant() {
    let (handle, mut scheduler) = ActorScheduler::new(10, 100);
    let mut log = LaneLog::default();
    let now = Instant::now();

    Application::send(
        &handle,
        EngineEvent::Data(EngineEventData::RequestFrame {
            timestamp: now,
            target_timestamp: now + Duration::from_millis(16),
            refresh_interval: Duration::from_millis(16),
        }),
    )
    .expect("data event should be accepted");
    Application::send(
        &handle,
        EngineEvent::Management(EngineEventManagement::KeyDown {
            key: KeySymbol::Enter,
            mods: Modifiers::empty(),
            text: None,
        }),
    )
    .expect("management event should be accepted");
    Application::send(
        &handle,
        EngineEvent::Control(EngineEventControl::ScaleChanged {
            id: WindowId::PRIMARY,
            scale: 2.0,
        }),
    )
    .expect("control event should be accepted");

    while log.lanes.len() < 3 {
        assert!(
            !scheduler.poll_once(&mut log),
            "the scheduler exited before delivering all three events"
        );
    }

    // Priority drain: control first, then management, then data.
    assert_eq!(log.lanes, vec!["control", "management", "data"]);
}
