#[cfg(target_os = "macos")]
mod tests {
    use actor_scheduler::{Actor, ActorStatus, HandlerError, HandlerResult, SystemStatus};
    use pixelflow_runtime::api::private::{create_engine_actor, EngineControl, EngineData};
    use pixelflow_runtime::api::public::{AppManagement, WindowDescriptor};
    use pixelflow_runtime::display::messages::{DisplayControl, DisplayMgmt};
    use pixelflow_runtime::display::ops::PlatformOps;
    use pixelflow_runtime::platform::macos::MetalOps;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    // A mock Engine actor to capture events
    struct MockEngine {
        pub captured_events: Arc<Mutex<Vec<pixelflow_runtime::display::messages::DisplayEvent>>>,
    }

    impl MockEngine {
        fn new(
            captured_events: Arc<Mutex<Vec<pixelflow_runtime::display::messages::DisplayEvent>>>,
        ) -> Self {
            Self { captured_events }
        }
    }

    // Use generics as required by the Actor trait definition
    impl Actor<EngineData, EngineControl, AppManagement> for MockEngine {
        fn handle_data(&mut self, msg: EngineData) -> HandlerResult {
            if let EngineData::FromDriver(evt) = msg {
                self.captured_events.lock().unwrap().push(evt);
            }
            Ok(())
        }

        fn handle_control(&mut self, _msg: EngineControl) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _msg: AppManagement) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    #[test]
    #[ignore = "Requires UI interaction or window server"]
    fn test_metal_ops_lifecycle() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_clone = events.clone();

        // 1. Create Engine Actor (Scheduler + Handle)
        let (handle, mut scheduler) = create_engine_actor(None);

        // 2. Spawn Scheduler in background
        thread::spawn(move || {
            let mut mock_engine = MockEngine::new(events_clone);
            scheduler.run(&mut mock_engine);
        });

        // 3. Instantiate MetalOps
        let mut ops = MetalOps::new(handle).expect("Failed to create MetalOps");

        // 4. Create a Window
        let settings = WindowDescriptor {
            title: "Integration Test Window".to_string(),
            width: 800,
            height: 600,
            ..Default::default()
        };

        let _ = ops.handle_management(DisplayMgmt::Create { settings });

        // 5. Emulate run loop step (Platform)
        // This should trigger window creation and send event to Engine
        let _ = ops.park(SystemStatus::Busy);

        // Give some time for message passing
        thread::sleep(Duration::from_millis(100));

        // 6. Verify Window Creation Event within MockEngine
        let captured = events.lock().unwrap();
        let found_window_id = captured.iter().find_map(|e| {
            if let pixelflow_runtime::display::messages::DisplayEvent::WindowCreated { window } = e
            {
                Some(window.id)
            } else {
                None
            }
        });
        assert!(
            found_window_id.is_some(),
            "Expected WindowCreated event, found: {:?}",
            *captured
        );

        // 7. Update Window Title
        let win_id = found_window_id.unwrap();
        let _ = ops.handle_control(DisplayControl::SetTitle {
            id: win_id,
            title: "Updated Title".to_string(),
        });

        // 8. Explicitly drop ops to close the handle, allowing scheduler to exit (though thread detach is fine for test)
        drop(ops);
    }
}
