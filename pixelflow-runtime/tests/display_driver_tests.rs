//! Display driver tests
//!
//! Tests for display message types, window descriptors, and input handling.

use pixelflow_runtime::api::public::{
    AppManagement, CursorIcon, EngineEvent, EngineEventControl, EngineEventData,
    EngineEventManagement, WindowDescriptor,
};
use pixelflow_runtime::display::messages::{DisplayControl, DisplayEvent, DisplayMgmt, WindowId};
use pixelflow_runtime::input::{KeySymbol, Modifiers, MouseButton};

// ============================================================================
// DisplayControl Tests
// ============================================================================

#[test]
fn display_control_debug_format() {
    let ctrl = DisplayControl::Bell;
    let debug_str = format!("{:?}", ctrl);
    assert!(debug_str.contains("Bell"));
}

#[test]
fn display_control_clone_works() {
    let ctrl1 = DisplayControl::SetTitle {
        id: WindowId::PRIMARY,
        title: "Test Window".to_string(),
    };
    let ctrl2 = ctrl1.clone();
    if let DisplayControl::SetTitle { title, .. } = ctrl2 {
        assert_eq!(title, "Test Window");
    } else {
        panic!("Clone did not preserve variant");
    }
}

// ============================================================================
// DisplayEvent Tests
// ============================================================================

#[test]
fn display_event_window_created_carries_dimensions() {
    use pixelflow_graphics::render::Frame;
    use pixelflow_runtime::display::messages::Window;
    use pixelflow_runtime::pixel::PlatformPixel;

    let event = DisplayEvent::WindowCreated {
        window: Window {
            id: WindowId::PRIMARY,
            width_px: 1920,
            height_px: 1080,
            scale: 2.0,
            frame: Frame::<PlatformPixel>::new(1920, 1080),
        },
    };

    if let DisplayEvent::WindowCreated { window } = event {
        assert_eq!(window.width_px, 1920);
        assert_eq!(window.height_px, 1080);
        assert!((window.scale - 2.0).abs() < 0.001);
    } else {
        panic!("Wrong variant");
    }
}

#[test]
fn display_event_key_event_with_modifiers() {
    let event = DisplayEvent::Key {
        id: WindowId::PRIMARY,
        symbol: KeySymbol::Char('a'),
        modifiers: Modifiers::CONTROL | Modifiers::SHIFT,
        text: Some("A".to_string()),
    };

    if let DisplayEvent::Key {
        symbol,
        modifiers,
        text,
        ..
    } = event
    {
        assert_eq!(symbol, KeySymbol::Char('a'));
        assert!(modifiers.contains(Modifiers::CONTROL));
        assert!(modifiers.contains(Modifiers::SHIFT));
        assert_eq!(text, Some("A".to_string()));
    } else {
        panic!("Wrong variant");
    }
}

#[test]
fn display_event_mouse_events() {
    let press_event = DisplayEvent::MouseButtonPress {
        id: WindowId::PRIMARY,
        button: 1,
        x: 100,
        y: 200,
        modifiers: Modifiers::empty(),
    };

    if let DisplayEvent::MouseButtonPress { button, x, y, .. } = press_event {
        assert_eq!(button, 1);
        assert_eq!(x, 100);
        assert_eq!(y, 200);
    } else {
        panic!("Wrong variant");
    }
}

#[test]
fn display_event_scroll_with_float_deltas() {
    let scroll = DisplayEvent::MouseScroll {
        id: WindowId::PRIMARY,
        dx: 0.5,
        dy: -1.5,
        x: 50,
        y: 75,
        modifiers: Modifiers::ALT,
    };

    if let DisplayEvent::MouseScroll {
        dx, dy, modifiers, ..
    } = scroll
    {
        assert!((dx - 0.5).abs() < 0.001);
        assert!((dy - (-1.5)).abs() < 0.001);
        assert!(modifiers.contains(Modifiers::ALT));
    } else {
        panic!("Wrong variant");
    }
}

#[test]
fn display_event_paste_data_works() {
    let event = DisplayEvent::PasteData {
        text: "Hello, clipboard!".to_string(),
    };
    if let DisplayEvent::PasteData { text } = event {
        assert_eq!(text, "Hello, clipboard!");
    } else {
        panic!("PasteData variant check failed");
    }
}

// ============================================================================
// DisplayMgmt Tests
// ============================================================================

#[test]
fn display_mgmt_create_window() {
    let mgmt = DisplayMgmt::Create {
        settings: WindowDescriptor::default(),
    };

    if let DisplayMgmt::Create { settings, .. } = mgmt {
        assert_eq!(settings.width, 800);
        assert_eq!(settings.height, 600);
    } else {
        panic!("Wrong variant");
    }
}

// ============================================================================
// WindowDescriptor Tests
// ============================================================================

#[test]
fn window_descriptor_default_values() {
    let desc = WindowDescriptor::default();
    assert_eq!(desc.width, 800);
    assert_eq!(desc.height, 600);
    assert_eq!(desc.title, "PixelFlow");
    assert!(desc.resizable);
}

#[test]
fn window_descriptor_custom_values() {
    let desc = WindowDescriptor {
        width: 1920,
        height: 1080,
        title: "My App".to_string(),
        resizable: false,
    };
    assert_eq!(desc.width, 1920);
    assert_eq!(desc.height, 1080);
    assert_eq!(desc.title, "My App");
    assert!(!desc.resizable);
}

// ============================================================================
// KeySymbol Tests
// ============================================================================

#[test]
fn key_symbol_modifier_detection() {
    assert!(KeySymbol::Shift.is_modifier());
    assert!(KeySymbol::Control.is_modifier());
    assert!(KeySymbol::Alt.is_modifier());
    assert!(KeySymbol::Super.is_modifier());
    assert!(KeySymbol::CapsLock.is_modifier());
    assert!(KeySymbol::NumLock.is_modifier());

    assert!(!KeySymbol::Enter.is_modifier());
    assert!(!KeySymbol::Char('a').is_modifier());
    assert!(!KeySymbol::F1.is_modifier());
}

#[test]
fn key_symbol_default_is_unknown() {
    let key: KeySymbol = Default::default();
    assert_eq!(key, KeySymbol::Unknown);
}

#[test]
fn key_symbol_equality() {
    assert_eq!(KeySymbol::Char('a'), KeySymbol::Char('a'));
    assert_ne!(KeySymbol::Char('a'), KeySymbol::Char('b'));
    assert_ne!(KeySymbol::Enter, KeySymbol::Tab);
}

// ============================================================================
// Modifiers Tests
// ============================================================================

#[test]
fn modifiers_default_is_empty() {
    let mods: Modifiers = Default::default();
    assert!(mods.is_empty());
}

#[test]
fn modifiers_can_combine() {
    let mods = Modifiers::CONTROL | Modifiers::SHIFT | Modifiers::ALT;
    assert!(mods.contains(Modifiers::CONTROL));
    assert!(mods.contains(Modifiers::SHIFT));
    assert!(mods.contains(Modifiers::ALT));
    assert!(!mods.contains(Modifiers::SUPER));
}

#[test]
fn modifiers_clone_and_copy() {
    let mods1 = Modifiers::CONTROL | Modifiers::SUPER;
    let mods2 = mods1; // Copy
    let mods3 = mods1; // Clone
    assert_eq!(mods1, mods2);
    assert_eq!(mods1, mods3);
}

// ============================================================================
// MouseButton Tests
// ============================================================================

#[test]
fn mouse_button_variants() {
    assert_eq!(MouseButton::Left, MouseButton::Left);
    assert_ne!(MouseButton::Left, MouseButton::Right);
    assert_eq!(MouseButton::Other(4), MouseButton::Other(4));
    assert_ne!(MouseButton::Other(4), MouseButton::Other(5));
}

// ============================================================================
// CursorIcon Tests
// ============================================================================

#[test]
fn cursor_icon_variants() {
    let default = CursorIcon::Default;
    let pointer = CursorIcon::Pointer;
    let text = CursorIcon::Text;

    assert!(matches!(default, CursorIcon::Default));
    assert!(matches!(pointer, CursorIcon::Pointer));
    assert!(matches!(text, CursorIcon::Text));
}

// ============================================================================
// EngineEvent Tests
// ============================================================================

#[test]
fn engine_event_control_resize() {
    let event = EngineEvent::Control(EngineEventControl::Resized {
        id: WindowId(0),
        width_px: 1920,
        height_px: 1080,
    });
    if let EngineEvent::Control(EngineEventControl::Resized {
        width_px: w,
        height_px: h,
        ..
    }) = event
    {
        assert_eq!(w, 1920);
        assert_eq!(h, 1080);
    } else {
        panic!("Wrong variant");
    }
}

#[test]
fn engine_event_control_scale_changed() {
    let event = EngineEvent::Control(EngineEventControl::ScaleChanged {
        id: WindowId(0),
        scale: 2.0,
    });
    if let EngineEvent::Control(EngineEventControl::ScaleChanged { scale, .. }) = event {
        assert!((scale - 2.0).abs() < 0.001);
    } else {
        panic!("Wrong variant");
    }
}

#[test]
fn engine_event_management_key_down() {
    let event = EngineEvent::Management(EngineEventManagement::KeyDown {
        key: KeySymbol::Enter,
        mods: Modifiers::CONTROL,
        text: None,
    });

    if let EngineEvent::Management(EngineEventManagement::KeyDown { key, mods, text }) = event {
        assert_eq!(key, KeySymbol::Enter);
        assert!(mods.contains(Modifiers::CONTROL));
        assert!(text.is_none());
    } else {
        panic!("Wrong variant");
    }
}

#[test]
fn engine_event_data_request_frame() {
    use std::time::{Duration, Instant};

    let now = Instant::now();
    let event = EngineEvent::Data(EngineEventData::RequestFrame {
        timestamp: now,
        target_timestamp: now + Duration::from_millis(16),
        refresh_interval: Duration::from_millis(16),
    });

    if let EngineEvent::Data(EngineEventData::RequestFrame {
        refresh_interval, ..
    }) = event
    {
        assert_eq!(refresh_interval, Duration::from_millis(16));
    } else {
        panic!("Wrong variant");
    }
}

// ============================================================================
// AppManagement Tests
// ============================================================================

#[test]
fn app_management_set_title() {
    let cmd = AppManagement::SetTitle("New Title".to_string());
    if let AppManagement::SetTitle(title) = cmd {
        assert_eq!(title, "New Title");
    } else {
        panic!("Wrong variant");
    }
}

#[test]
fn app_management_resize_request() {
    let cmd = AppManagement::ResizeRequest(1280, 720);
    if let AppManagement::ResizeRequest(w, h) = cmd {
        assert_eq!(w, 1280);
        assert_eq!(h, 720);
    } else {
        panic!("Wrong variant");
    }
}

#[test]
fn app_management_clipboard() {
    let cmd = AppManagement::CopyToClipboard("test text".to_string());
    if let AppManagement::CopyToClipboard(text) = cmd {
        assert_eq!(text, "test text");
    } else {
        panic!("Wrong variant");
    }
}

#[test]
fn app_management_cursor_icon() {
    let cmd = AppManagement::SetCursorIcon(CursorIcon::Text);
    if let AppManagement::SetCursorIcon(icon) = cmd {
        assert!(matches!(icon, CursorIcon::Text));
    } else {
        panic!("Wrong variant");
    }
}
