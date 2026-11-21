//! Behavior of the public input and window types.
//!
//! Only logic a caller relies on is tested here: the documented defaults and the
//! modifier-key classification. Constructing a message and destructuring it again
//! asserts nothing the compiler does not already guarantee.

use pixelflow_runtime::api::public::WindowDescriptor;
use pixelflow_runtime::input::{KeySymbol, Modifiers};

#[test]
fn window_descriptor_default_opens_a_resizable_800_by_600_pixelflow_window() {
    let desc = WindowDescriptor::default();
    assert_eq!(desc.width, 800);
    assert_eq!(desc.height, 600);
    assert_eq!(desc.title, "PixelFlow");
    assert!(desc.resizable);
}

#[test]
fn key_symbol_reports_shift_control_alt_super_and_lock_keys_as_modifiers() {
    for key in [
        KeySymbol::Shift,
        KeySymbol::Control,
        KeySymbol::Alt,
        KeySymbol::Super,
        KeySymbol::CapsLock,
        KeySymbol::NumLock,
    ] {
        assert!(key.is_modifier(), "{key:?} should be a modifier");
    }
}

#[test]
fn key_symbol_does_not_report_characters_or_function_keys_as_modifiers() {
    for key in [KeySymbol::Enter, KeySymbol::Char('a'), KeySymbol::F1] {
        assert!(!key.is_modifier(), "{key:?} should not be a modifier");
    }
}

#[test]
fn key_symbol_defaults_to_unknown() {
    assert_eq!(KeySymbol::default(), KeySymbol::Unknown);
}

#[test]
fn modifiers_default_to_no_modifier_held() {
    assert!(Modifiers::default().is_empty());
}
