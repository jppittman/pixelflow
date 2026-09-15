// src/term/emulator/mouse.rs

//! Mouse event encoding for terminal mouse tracking protocols.
//!
//! When a shell application enables mouse tracking (via DEC private modes like
//! 1000, 1002, 1003), mouse events must be encoded as escape sequences and
//! sent to the PTY. This module handles the encoding for both SGR (mode 1006)
//! and legacy X10/Normal mouse protocols.

use crate::term::modes::DecPrivateModes;
use pixelflow_runtime::input::MouseButton;
use std::io::Write;

/// The type of mouse event being reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseEventKind {
    Press,
    Release,
    Motion,
}

/// Parameters for encoding a mouse event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MouseEncodingParams {
    pub button: MouseButton,
    pub col: usize,
    pub row: usize,
    pub kind: MouseEventKind,
}

/// Encode a mouse event as terminal escape sequence bytes.
///
/// Returns `None` if no mouse tracking mode is active, or if the current mode
/// doesn't report this event kind (e.g., X10 mode doesn't report releases).
///
/// Coordinates `col` and `row` are 0-based cell positions.
pub(crate) fn encode_mouse_event(
    modes: &DecPrivateModes,
    params: MouseEncodingParams,
) -> Option<Vec<u8>> {
    // Determine if the current tracking mode reports this event kind
    if !should_report(modes, params.kind) {
        return None;
    }

    if modes.mouse_sgr_mode {
        let button_code = sgr_button_code(params.button, params.kind);
        Some(encode_sgr(button_code, params.col, params.row, params.kind))
    } else {
        let button_code = legacy_button_code(params.button, params.kind);
        encode_legacy(button_code, params.col, params.row)
    }
}

/// Check whether the active tracking mode should report this event kind.
fn should_report(modes: &DecPrivateModes, kind: MouseEventKind) -> bool {
    match kind {
        MouseEventKind::Press => {
            modes.mouse_x10_mode
                || modes.mouse_vt200_mode
                || modes.mouse_button_event_mode
                || modes.mouse_any_event_mode
        }
        MouseEventKind::Release => {
            // X10 mode does not report releases
            modes.mouse_vt200_mode || modes.mouse_button_event_mode || modes.mouse_any_event_mode
        }
        MouseEventKind::Motion => {
            // Button-event mode reports motion only while a button is held,
            // but filtering by held-button state is done by the caller.
            // Any-event mode reports all motion.
            modes.mouse_button_event_mode || modes.mouse_any_event_mode
        }
    }
}

/// Map a button to its base code for the xterm protocol.
///
/// Base codes: 0=left, 1=middle, 2=right, 64=scroll_up, 65=scroll_down.
fn button_base_code(button: MouseButton) -> u8 {
    match button {
        MouseButton::Left => 0,
        MouseButton::Middle => 1,
        MouseButton::Right => 2,
        MouseButton::ScrollUp => 64,
        MouseButton::ScrollDown => 65,
        MouseButton::Other(n) => {
            // Buttons 4+ map to codes 128+
            if n >= 4 {
                128 + n - 4
            } else {
                n
            }
        }
    }
}

/// Compute the button code for SGR encoding.
///
/// SGR distinguishes press from release via the suffix character (M vs m),
/// so the button code always carries the actual button identity.
/// Motion events add 32 to the code.
fn sgr_button_code(button: MouseButton, kind: MouseEventKind) -> u8 {
    let base = button_base_code(button);
    if kind == MouseEventKind::Motion {
        base + 32
    } else {
        base
    }
}

/// Compute the button code for legacy (X10/Normal) encoding.
///
/// In legacy mode, release events use code 3 (no button identity) because
/// the protocol has no other way to signal a release. Motion events add 32.
fn legacy_button_code(button: MouseButton, kind: MouseEventKind) -> u8 {
    match kind {
        MouseEventKind::Release => 3,
        MouseEventKind::Motion => button_base_code(button) + 32,
        MouseEventKind::Press => button_base_code(button),
    }
}

/// Encode using SGR extended mouse mode (1006).
///
/// Format: `ESC [ < Cb ; Cx ; Cy M` for press/motion, `ESC [ < Cb ; Cx ; Cy m` for release.
/// Coordinates are 1-based.
fn encode_sgr(button_code: u8, col: usize, row: usize, kind: MouseEventKind) -> Vec<u8> {
    let suffix = if kind == MouseEventKind::Release {
        b'm'
    } else {
        b'M'
    };
    // SGR uses 1-based coordinates
    let cx = col + 1;
    let cy = row + 1;
    // Max realistic: "\x1b[<999;99999;99999M" = ~22 bytes
    let mut buf = Vec::with_capacity(24);
    write!(buf, "\x1b[<{};{};{}", button_code, cx, cy).unwrap();
    buf.push(suffix);
    buf
}

/// Encode using legacy X10/Normal mouse mode.
///
/// Format: `ESC [ M Cb Cx Cy` where each of Cb, Cx, Cy is a single byte + 32.
/// Limited to coordinates 0-222 (encoded as 32-254).
/// Returns `None` if coordinates exceed the encodable range.
fn encode_legacy(button_code: u8, col: usize, row: usize) -> Option<Vec<u8>> {
    // Legacy encoding caps at 222 (byte value 254, since we add 32)
    if col > 222 || row > 222 {
        return None;
    }
    let cb = button_code + 32;
    let cx = (col as u8) + 33; // 1-based + 32
    let cy = (row as u8) + 33;
    Some(vec![b'\x1b', b'[', b'M', cb, cx, cy])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn modes_with_vt200() -> DecPrivateModes {
        DecPrivateModes {
            mouse_vt200_mode: true,
            ..Default::default()
        }
    }

    fn modes_with_sgr() -> DecPrivateModes {
        DecPrivateModes {
            mouse_vt200_mode: true,
            mouse_sgr_mode: true,
            ..Default::default()
        }
    }

    fn modes_with_x10() -> DecPrivateModes {
        DecPrivateModes {
            mouse_x10_mode: true,
            ..Default::default()
        }
    }

    fn modes_with_any_event_sgr() -> DecPrivateModes {
        DecPrivateModes {
            mouse_any_event_mode: true,
            mouse_sgr_mode: true,
            ..Default::default()
        }
    }

    fn modes_with_button_event_sgr() -> DecPrivateModes {
        DecPrivateModes {
            mouse_button_event_mode: true,
            mouse_sgr_mode: true,
            ..Default::default()
        }
    }

    #[test]
    fn it_should_return_none_when_no_mouse_tracking_mode_is_active() {
        let modes = DecPrivateModes::default();
        let result = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 5,
                row: 10,
                kind: MouseEventKind::Press,
            },
        );
        assert_eq!(result, None);
    }

    #[test]
    fn it_should_encode_sgr_left_button_press_with_1_based_coordinates() {
        let modes = modes_with_sgr();
        let result = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 5,
                row: 10,
                kind: MouseEventKind::Press,
            },
        )
        .unwrap();
        // SGR: ESC[<0;6;11M (1-based coords)
        assert_eq!(result, b"\x1b[<0;6;11M");
    }

    #[test]
    fn it_should_encode_sgr_left_button_release_with_a_lowercase_m_suffix() {
        let modes = modes_with_sgr();
        let result = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 5,
                row: 10,
                kind: MouseEventKind::Release,
            },
        )
        .unwrap();
        // SGR release uses lowercase 'm', button code preserved
        assert_eq!(result, b"\x1b[<0;6;11m");
    }

    #[test]
    fn it_should_encode_sgr_right_button_press_at_the_origin() {
        let modes = modes_with_sgr();
        let result = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Right,
                col: 0,
                row: 0,
                kind: MouseEventKind::Press,
            },
        )
        .unwrap();
        assert_eq!(result, b"\x1b[<2;1;1M");
    }

    #[test]
    fn it_should_preserve_button_identity_on_sgr_release() {
        let modes = modes_with_sgr();
        let result = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Right,
                col: 3,
                row: 7,
                kind: MouseEventKind::Release,
            },
        )
        .unwrap();
        // SGR preserves button identity on release
        assert_eq!(result, b"\x1b[<2;4;8m");
    }

    #[test]
    fn it_should_encode_sgr_middle_button_press() {
        let modes = modes_with_sgr();
        let result = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Middle,
                col: 79,
                row: 23,
                kind: MouseEventKind::Press,
            },
        )
        .unwrap();
        assert_eq!(result, b"\x1b[<1;80;24M");
    }

    #[test]
    fn it_should_add_32_to_the_button_code_for_sgr_motion_in_any_event_mode() {
        let modes = modes_with_any_event_sgr();
        let result = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 10,
                row: 5,
                kind: MouseEventKind::Motion,
            },
        )
        .unwrap();
        // Motion adds 32 to button code: 0 + 32 = 32
        assert_eq!(result, b"\x1b[<32;11;6M");
    }

    #[test]
    fn it_should_encode_sgr_motion_when_only_button_event_mode_is_enabled() {
        let modes = modes_with_button_event_sgr();
        let result = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Right,
                col: 10,
                row: 5,
                kind: MouseEventKind::Motion,
            },
        )
        .unwrap();
        // Motion adds 32 to button code: 2 + 32 = 34
        assert_eq!(result, b"\x1b[<34;11;6M");
    }

    #[test]
    fn it_should_encode_sgr_scroll_up_as_button_code_64() {
        let modes = modes_with_sgr();
        let result = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::ScrollUp,
                col: 10,
                row: 5,
                kind: MouseEventKind::Press,
            },
        )
        .unwrap();
        assert_eq!(result, b"\x1b[<64;11;6M");
    }

    #[test]
    fn it_should_encode_sgr_scroll_down_as_button_code_65() {
        let modes = modes_with_sgr();
        let result = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::ScrollDown,
                col: 10,
                row: 5,
                kind: MouseEventKind::Press,
            },
        )
        .unwrap();
        assert_eq!(result, b"\x1b[<65;11;6M");
    }

    #[test]
    fn it_should_encode_legacy_left_button_press() {
        let modes = modes_with_vt200();
        let result = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 5,
                row: 10,
                kind: MouseEventKind::Press,
            },
        )
        .unwrap();
        // Legacy: ESC[M + (0+32) + (5+33) + (10+33)
        assert_eq!(result, vec![0x1b, b'[', b'M', 32, 38, 43]);
    }

    #[test]
    fn it_should_encode_legacy_release_using_button_code_3() {
        let modes = modes_with_vt200();
        let result = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 5,
                row: 10,
                kind: MouseEventKind::Release,
            },
        )
        .unwrap();
        // Legacy release: button code = 3, so Cb = 3 + 32 = 35
        assert_eq!(result, vec![0x1b, b'[', b'M', 35, 38, 43]);
    }

    #[test]
    fn it_should_use_button_code_3_for_legacy_release_regardless_of_which_button_was_released() {
        let modes = modes_with_vt200();
        let result = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Right,
                col: 5,
                row: 10,
                kind: MouseEventKind::Release,
            },
        )
        .unwrap();
        // Legacy release always uses code 3 regardless of which button was released
        assert_eq!(result, vec![0x1b, b'[', b'M', 35, 38, 43]);
    }

    #[test]
    fn it_should_return_none_when_legacy_coordinates_exceed_222() {
        let modes = modes_with_vt200();
        let result = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 300,
                row: 10,
                kind: MouseEventKind::Press,
            },
        );
        assert_eq!(result, None);
    }

    #[test]
    fn it_should_encode_only_press_events_in_x10_mouse_mode() {
        let modes = modes_with_x10();
        let press = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 0,
                row: 0,
                kind: MouseEventKind::Press,
            },
        );
        assert!(press.is_some());
        let release = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 0,
                row: 0,
                kind: MouseEventKind::Release,
            },
        );
        assert_eq!(release, None);
        let motion = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 0,
                row: 0,
                kind: MouseEventKind::Motion,
            },
        );
        assert_eq!(motion, None);
    }

    #[test]
    fn it_should_encode_both_press_and_release_events_in_vt200_mouse_mode() {
        let modes = modes_with_vt200();
        let press = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 0,
                row: 0,
                kind: MouseEventKind::Press,
            },
        );
        assert!(press.is_some());
        let release = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 0,
                row: 0,
                kind: MouseEventKind::Release,
            },
        );
        assert!(release.is_some());
        let motion = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 0,
                row: 0,
                kind: MouseEventKind::Motion,
            },
        );
        assert_eq!(motion, None);
    }

    #[test]
    fn it_should_encode_sgr_coordinates_beyond_the_legacy_222_limit() {
        let modes = modes_with_sgr();
        // SGR has no coordinate limit
        let result = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 500,
                row: 300,
                kind: MouseEventKind::Press,
            },
        )
        .unwrap();
        assert_eq!(result, b"\x1b[<0;501;301M");
    }

    fn modes_with_button_event_legacy() -> DecPrivateModes {
        DecPrivateModes {
            mouse_button_event_mode: true,
            ..Default::default()
        }
    }

    fn modes_with_any_event_legacy() -> DecPrivateModes {
        DecPrivateModes {
            mouse_any_event_mode: true,
            ..Default::default()
        }
    }

    #[test]
    fn it_should_report_press_events_when_only_button_event_mode_is_enabled() {
        let modes = modes_with_button_event_legacy();
        let result = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 0,
                row: 0,
                kind: MouseEventKind::Press,
            },
        );
        assert!(result.is_some());
    }

    #[test]
    fn it_should_report_release_events_when_only_button_event_mode_is_enabled() {
        let modes = modes_with_button_event_legacy();
        let result = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 0,
                row: 0,
                kind: MouseEventKind::Release,
            },
        );
        assert!(result.is_some());
    }

    #[test]
    fn it_should_add_32_to_the_button_code_for_legacy_motion_events() {
        let modes = modes_with_any_event_legacy();
        let result = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 5,
                row: 10,
                kind: MouseEventKind::Motion,
            },
        )
        .unwrap();
        // base(Left)=0, +32 for motion = 32, then +32 legacy offset = 64
        assert_eq!(result, vec![0x1b, b'[', b'M', 64, 38, 43]);
    }

    #[test]
    fn it_should_use_the_raw_button_number_for_other_buttons_below_four() {
        let modes = modes_with_sgr();
        let result = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Other(3),
                col: 0,
                row: 0,
                kind: MouseEventKind::Press,
            },
        )
        .unwrap();
        assert_eq!(result, b"\x1b[<3;1;1M");
    }

    #[test]
    fn it_should_map_other_buttons_at_or_above_four_using_128_plus_n_minus_4() {
        let modes = modes_with_sgr();
        let boundary = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Other(4),
                col: 0,
                row: 0,
                kind: MouseEventKind::Press,
            },
        )
        .unwrap();
        assert_eq!(boundary, b"\x1b[<128;1;1M");

        let above = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Other(8),
                col: 0,
                row: 0,
                kind: MouseEventKind::Press,
            },
        )
        .unwrap();
        assert_eq!(above, b"\x1b[<132;1;1M");
    }

    #[test]
    fn it_should_accept_column_222_but_reject_223_in_legacy_encoding() {
        let modes = modes_with_vt200();
        let at_boundary = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 222,
                row: 0,
                kind: MouseEventKind::Press,
            },
        );
        assert!(at_boundary.is_some());

        let over_boundary = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 223,
                row: 0,
                kind: MouseEventKind::Press,
            },
        );
        assert_eq!(over_boundary, None);
    }

    #[test]
    fn it_should_accept_row_222_but_reject_223_in_legacy_encoding() {
        let modes = modes_with_vt200();
        let at_boundary = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 0,
                row: 222,
                kind: MouseEventKind::Press,
            },
        );
        assert!(at_boundary.is_some());

        let over_boundary = encode_mouse_event(
            &modes,
            MouseEncodingParams {
                button: MouseButton::Left,
                col: 0,
                row: 223,
                kind: MouseEventKind::Press,
            },
        );
        assert_eq!(over_boundary, None);
    }
}
