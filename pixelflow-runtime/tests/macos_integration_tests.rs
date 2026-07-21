#[cfg(target_os = "macos")]
mod tests {
    use actor_scheduler::SystemStatus;
    use pixelflow_runtime::api::private::EngineData;
    use pixelflow_runtime::api::public::WindowDescriptor;
    use pixelflow_runtime::display::messages::{DisplayControl, DisplayEvent, DisplayMgmt};
    use pixelflow_runtime::display::ops::PlatformOps;
    use pixelflow_runtime::platform::macos::MetalOps;
    use pixelflow_runtime::testing::{MockEngine, ReceivedMessage};
    use std::thread;
    use std::time::Duration;

    #[test]
    #[ignore = "Requires UI interaction or window server"]
    fn metal_ops_lifecycle() {
        // The crate's sanctioned test double already runs an actor that
        // records every message sent to it and hands out a producer handle -
        // no need to hand-roll another Actor impl against api::private types.
        let mut mock_engine = MockEngine::new();
        let handle = mock_engine.take_handle();

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
        let captured = mock_engine.messages();
        let found_window_id = captured.iter().find_map(|msg| {
            let ReceivedMessage::Data(EngineData::FromDriver(DisplayEvent::WindowCreated {
                window,
            })) = msg
            else {
                return None;
            };
            Some(window.id)
        });
        assert!(
            found_window_id.is_some(),
            "Expected WindowCreated event, found: {:?}",
            *captured
        );
        drop(captured);

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
