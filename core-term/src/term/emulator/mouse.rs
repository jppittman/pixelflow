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
            modes.mouse_vt200_mode
                || modes.mouse_button_event_mode
                || modes.mouse_any_event_mode
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
            if n >= 4 { 128 + n - 4 } else { n }
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
    let suffix = if kind == MouseEventKind::Release { b'm' } else { b'M' };
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
        let mut modes = DecPrivateModes::default();
        modes.mouse_vt200_mode = true;
        modes
    }

    fn modes_with_sgr() -> DecPrivateModes {
        let mut modes = DecPrivateModes::default();
        modes.mouse_vt200_mode = true;
        modes.mouse_sgr_mode = true;
        modes
    }

    fn modes_with_x10() -> DecPrivateModes {
        let mut modes = DecPrivateModes::default();
        modes.mouse_x10_mode = true;
        modes
    }

    fn modes_with_any_event_sgr() -> DecPrivateModes {
        let mut modes = DecPrivateModes::default();
        modes.mouse_any_event_mode = true;
        modes.mouse_sgr_mode = true;
        modes
    }

    fn modes_with_button_event_sgr() -> DecPrivateModes {
        let mut modes = DecPrivateModes::default();
        modes.mouse_button_event_mode = true;
        modes.mouse_sgr_mode = true;
        modes
    }

    #[test]
    fn no_mode_returns_none() {
        let modes = DecPrivateModes::default();
        let result = encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::Left, col: 5, row: 10, kind: MouseEventKind::Press });
        assert_eq!(result, None);
    }

    #[test]
    fn sgr_left_press() {
        let modes = modes_with_sgr();
        let result =
            encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::Left, col: 5, row: 10, kind: MouseEventKind::Press }).unwrap();
        // SGR: ESC[<0;6;11M (1-based coords)
        assert_eq!(result, b"\x1b[<0;6;11M");
    }

    #[test]
    fn sgr_left_release() {
        let modes = modes_with_sgr();
        let result =
            encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::Left, col: 5, row: 10, kind: MouseEventKind::Release }).unwrap();
        // SGR release uses lowercase 'm', button code preserved
        assert_eq!(result, b"\x1b[<0;6;11m");
    }

    #[test]
    fn sgr_right_press() {
        let modes = modes_with_sgr();
        let result =
            encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::Right, col: 0, row: 0, kind: MouseEventKind::Press }).unwrap();
        assert_eq!(result, b"\x1b[<2;1;1M");
    }

    #[test]
    fn sgr_right_release() {
        let modes = modes_with_sgr();
        let result =
            encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::Right, col: 3, row: 7, kind: MouseEventKind::Release }).unwrap();
        // SGR preserves button identity on release
        assert_eq!(result, b"\x1b[<2;4;8m");
    }

    #[test]
    fn sgr_middle_press() {
        let modes = modes_with_sgr();
        let result =
            encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::Middle, col: 79, row: 23, kind: MouseEventKind::Press })
                .unwrap();
        assert_eq!(result, b"\x1b[<1;80;24M");
    }

    #[test]
    fn sgr_motion_left() {
        let modes = modes_with_any_event_sgr();
        let result =
            encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::Left, col: 10, row: 5, kind: MouseEventKind::Motion }).unwrap();
        // Motion adds 32 to button code: 0 + 32 = 32
        assert_eq!(result, b"\x1b[<32;11;6M");
    }

    #[test]
    fn sgr_motion_right() {
        let modes = modes_with_button_event_sgr();
        let result =
            encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::Right, col: 10, row: 5, kind: MouseEventKind::Motion }).unwrap();
        // Motion adds 32 to button code: 2 + 32 = 34
        assert_eq!(result, b"\x1b[<34;11;6M");
    }

    #[test]
    fn sgr_scroll_up() {
        let modes = modes_with_sgr();
        let result =
            encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::ScrollUp, col: 10, row: 5, kind: MouseEventKind::Press })
                .unwrap();
        assert_eq!(result, b"\x1b[<64;11;6M");
    }

    #[test]
    fn sgr_scroll_down() {
        let modes = modes_with_sgr();
        let result =
            encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::ScrollDown, col: 10, row: 5, kind: MouseEventKind::Press })
                .unwrap();
        assert_eq!(result, b"\x1b[<65;11;6M");
    }

    #[test]
    fn legacy_left_press() {
        let modes = modes_with_vt200();
        let result =
            encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::Left, col: 5, row: 10, kind: MouseEventKind::Press }).unwrap();
        // Legacy: ESC[M + (0+32) + (5+33) + (10+33)
        assert_eq!(result, vec![0x1b, b'[', b'M', 32, 38, 43]);
    }

    #[test]
    fn legacy_left_release_uses_code_3() {
        let modes = modes_with_vt200();
        let result =
            encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::Left, col: 5, row: 10, kind: MouseEventKind::Release }).unwrap();
        // Legacy release: button code = 3, so Cb = 3 + 32 = 35
        assert_eq!(result, vec![0x1b, b'[', b'M', 35, 38, 43]);
    }

    #[test]
    fn legacy_right_release_uses_code_3() {
        let modes = modes_with_vt200();
        let result =
            encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::Right, col: 5, row: 10, kind: MouseEventKind::Release })
                .unwrap();
        // Legacy release always uses code 3 regardless of which button was released
        assert_eq!(result, vec![0x1b, b'[', b'M', 35, 38, 43]);
    }

    #[test]
    fn legacy_coords_overflow_returns_none() {
        let modes = modes_with_vt200();
        let result =
            encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::Left, col: 300, row: 10, kind: MouseEventKind::Press });
        assert_eq!(result, None);
    }

    #[test]
    fn x10_reports_press_only() {
        let modes = modes_with_x10();
        let press = encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::Left, col: 0, row: 0, kind: MouseEventKind::Press });
        assert!(press.is_some());
        let release =
            encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::Left, col: 0, row: 0, kind: MouseEventKind::Release });
        assert_eq!(release, None);
        let motion = encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::Left, col: 0, row: 0, kind: MouseEventKind::Motion });
        assert_eq!(motion, None);
    }

    #[test]
    fn vt200_reports_press_and_release() {
        let modes = modes_with_vt200();
        let press = encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::Left, col: 0, row: 0, kind: MouseEventKind::Press });
        assert!(press.is_some());
        let release =
            encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::Left, col: 0, row: 0, kind: MouseEventKind::Release });
        assert!(release.is_some());
        let motion = encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::Left, col: 0, row: 0, kind: MouseEventKind::Motion });
        assert_eq!(motion, None);
    }

    #[test]
    fn sgr_large_coordinates() {
        let modes = modes_with_sgr();
        // SGR has no coordinate limit
        let result =
            encode_mouse_event(&modes, MouseEncodingParams { button: MouseButton::Left, col: 500, row: 300, kind: MouseEventKind::Press })
                .unwrap();
        assert_eq!(result, b"\x1b[<0;501;301M");
    }
}
