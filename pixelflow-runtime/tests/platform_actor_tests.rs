use actor_scheduler::{Actor, ActorStatus, HandlerError, HandlerResult, SystemStatus};
use pixelflow_graphics::render::Frame;
use pixelflow_runtime::display::messages::{DisplayControl, DisplayData, DisplayMgmt};
use pixelflow_runtime::display::ops::PlatformOps;
use pixelflow_runtime::display::platform::PlatformActor;
use std::sync::{Arc, Mutex};

// Mock Platform Operations
#[derive(Clone)]
struct MockOps {
    // Shared state to verify calls
    pub log: Arc<Mutex<Vec<String>>>,
}

impl MockOps {
    fn new() -> Self {
        Self {
            log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn push_log(&self, msg: &str) {
        self.log.lock().unwrap().push(msg.to_string());
    }
}

unsafe impl Send for MockOps {}

impl PlatformOps for MockOps {
    fn handle_data(&mut self, msg: DisplayData) -> HandlerResult {
        match msg {
            DisplayData::Present { window } => {
                self.push_log(&format!("Present {:?}", window.id));
            }
        }
        Ok(())
    }

    fn handle_control(&mut self, msg: DisplayControl) -> HandlerResult {
        match msg {
            DisplayControl::SetTitle { id, title } => {
                self.push_log(&format!("SetTitle {:?} {}", id, title));
            }
            _ => self.push_log(&format!("Control {:?}", msg)),
        }
        Ok(())
    }

    fn handle_management(&mut self, msg: DisplayMgmt) -> HandlerResult {
        match msg {
            DisplayMgmt::Create { .. } => {
                self.push_log("Create");
            }
            _ => self.push_log(&format!("Management {:?}", msg)),
        }
        Ok(())
    }

    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        self.push_log("Park");
        Ok(ActorStatus::Idle)
    }
}

#[test]
fn platform_actor_delegation_works() {
    // 1. Create MockOps
    let ops = MockOps::new();
    let log_ref = ops.log.clone();

    // 2. Create PlatformActor
    let mut actor = PlatformActor::new(ops);

    // 3. Send messages manually to verify delegation
    // Note: We test `Actor` trait implementation directly, skipping Scheduler for unit test simplicity

    // Test Management (Create)
    actor
        .handle_management(DisplayMgmt::Create {
            settings: Default::default(),
        })
        .expect("handle_management should succeed");

    // Test Control (SetTitle)
    actor
        .handle_control(DisplayControl::SetTitle {
            id: pixelflow_runtime::api::private::WindowId(1),
            title: "Test Window".to_string(),
        })
        .expect("handle_control should succeed");

    // Test Data (Present)
    actor
        .handle_data(DisplayData::Present {
            window: pixelflow_runtime::display::messages::Window {
                id: pixelflow_runtime::api::private::WindowId(1),
                frame: Frame::new(100, 100),
                width_px: 100,
                height_px: 100,
                scale: 1.0,
            },
        })
        .expect("handle_data should succeed");

    // Test Park
    actor.park(SystemStatus::Busy).expect("park should succeed");

    // 4. Verify Log
    let log = log_ref.lock().unwrap();
    assert_eq!(log.len(), 4);
    assert_eq!(log[0], "Create");
    assert!(log[1].contains("SetTitle WindowId(1) Test Window"));
    assert!(log[2].contains("Present WindowId(1)"));
    assert_eq!(log[3], "Park");
}
