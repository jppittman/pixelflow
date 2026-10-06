// myterm/src/term/cursor.rs

//! Manages the terminal's cursor state, including its logical position,
//! attributes, visibility, and translation to physical screen coordinates.
//! This module aims to be the single source of truth for cursor-related information,
//! abstracting away the complexities of different terminal modes like Origin Mode (DECOM)
//! from the main terminal emulation logic.

use crate::term::cursor_visibility::CursorVisibility;
use crate::{config, glyph::Attributes};
use anyhow::{anyhow, Result};
use log::{trace, warn};
use serde::{
    de::{self, Deserializer, Visitor},
    Deserialize, Serialize, Serializer,
};
use std::{cmp::min, fmt};

// --- Structs ---
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum CursorShape {
    BlinkingBlock = 1,
    SteadyBlock = 2,
    BlinkingUnderline = 3,
    SteadyUnderline = 4,
    BlinkingBar = 5,
    SteadyBar = 6,
}

impl CursorShape {
    fn unfocused_default() -> Self {
        config::CONFIG.appearance.unfocused_cursor.shape
    }
    /// Creates a `CursorShape` from a u16 code as used in DECSCUSR.
    /// Handles unknown codes by defaulting and logging a warning.
    #[must_use]
    pub fn from_decscusr_code(code: u16) -> Self {
        match code {
            0 => CursorShape::default(),
            1 => CursorShape::BlinkingBlock,
            2 => CursorShape::SteadyBlock,
            3 => CursorShape::BlinkingUnderline,
            4 => CursorShape::SteadyUnderline,
            5 => CursorShape::BlinkingBar,
            6 => CursorShape::SteadyBar,
            _ => {
                warn!(
                    //
                    "Received unknown DECSCUSR shape code: {}. Defaulting to DefaultOrBlinkingBlock.",
                    code
                );
                CursorShape::default()
            }
        }
    }
    /// Returns the string representation for serialization.
    fn to_str(self) -> &'static str {
        match self {
            CursorShape::BlinkingBlock => "BlinkingBlock",
            CursorShape::SteadyBlock => "SteadyBlock",
            CursorShape::BlinkingUnderline => "BlinkingUnderline",
            CursorShape::SteadyUnderline => "SteadyUnderline",
            CursorShape::BlinkingBar => "BlinkingBar",
            CursorShape::SteadyBar => "SteadyBar",
        }
    }
    fn from_str(input: &str) -> Result<Self> {
        let val = match input {
            "BlinkingBlock" => CursorShape::BlinkingBlock,
            "SteadyBlock" => CursorShape::SteadyBlock,
            "BlinkingUnderline" => CursorShape::BlinkingUnderline,
            "SteadyUnderline" => CursorShape::SteadyUnderline,
            "BlinkingBar" => CursorShape::BlinkingBar,
            "SteadyBar" => CursorShape::SteadyBar,
            _ => return Err(anyhow!("unrecognized cursor shape, reverting to default")),
        };
        Ok(val)
    }
}

impl Default for CursorShape {
    fn default() -> Self {
        config::CONFIG.appearance.cursor.shape
    }
}

/// Represents the state of the terminal cursor.
///
/// This includes its logical position (which can be relative to scrolling margins
/// if origin mode is active), the SGR attributes for characters to be written
/// at the cursor, and its visibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cursor {
    /// Logical column (0-based). Can be == width to indicate next char causes wrap.
    pub logical_x: usize,
    /// Logical row (0-based).
    /// If origin mode is active, this is relative to the top of the scrolling region.
    /// Otherwise, it's relative to the top of the physical screen.
    pub logical_y: usize,
    /// Current SGR attributes for characters to be written at the cursor.
    pub attributes: Attributes,
    /// Visibility of the cursor. True if the cursor should be displayed.
    pub visible: bool,
    pub shape: CursorShape,
    pub unfocused_shape: CursorShape,
}

impl Default for Cursor {
    /// Creates a default `Cursor` instance.
    /// The cursor starts at logical (0,0), visible, and with default attributes.
    fn default() -> Self {
        Cursor {
            logical_x: 0,
            logical_y: 0,
            attributes: Attributes::default(), // Use attributes from a default glyph configuration.
            visible: true,
            shape: CursorShape::default(),
            unfocused_shape: CursorShape::unfocused_default(),
        }
    }
}

/// Provides the necessary screen context for cursor operations.
///
/// This includes the physical dimensions of the screen, the defined scrolling
/// region (if any), and whether Origin Mode (DECOM) is currently active.
/// This context is used by `CursorController` to correctly interpret logical
/// cursor movements and clamp positions to valid areas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScreenContext {
    /// Width of the screen in character cells.
    pub width: usize,
    /// Height of the screen in character cells.
    pub height: usize,
    /// Top row of the scrolling region (0-based, inclusive).
    pub scroll_top: usize,
    /// Bottom row of the scrolling region (0-based, inclusive).
    pub scroll_bot: usize,
    /// `true` if Origin Mode (DECOM) is active, meaning logical row 0
    /// corresponds to `scroll_top`. `false` means logical row 0 is physical row 0.
    pub origin_mode_active: bool,
}

/// Manages the terminal cursor's state and provides methods for logical movements.
///
/// The `CursorController` is responsible for:
/// - Maintaining the current logical position, SGR attributes, and visibility of the cursor.
/// - Translating logical cursor positions to physical screen positions based on the
///   provided `ScreenContext` (which includes information about origin mode and scrolling regions).
/// - Handling cursor saving and restoring (DECSC/DECRC).
/// - Clamping cursor movements to the valid boundaries defined by the `ScreenContext`.
#[derive(Debug, Clone)]
pub struct CursorController {
    /// The current state of the cursor.
    pub(super) cursor: Cursor,
    /// Saved cursor state for DECSC/DECRC functionality. `None` if no state is currently saved.
    pub(super) saved_cursor: Option<Cursor>,
}

// --- Implementations ---

impl CursorController {
    /// Creates a new `CursorController`.
    ///
    /// The cursor is initialized at logical position (0,0), visible, and with the
    /// specified `initial_attributes`.
    ///
    /// # Arguments
    /// * `initial_attributes` - The initial SGR attributes for the cursor.
    #[must_use]
    pub fn new(initial_attributes: Attributes) -> Self {
        trace!(
            "Creating new CursorController with initial attributes: {:?}",
            initial_attributes
        );
        Self {
            cursor: Cursor {
                attributes: initial_attributes,
                ..Default::default() // Sets logical_x=0, logical_y=0, visible=true
            },
            saved_cursor: None,
        }
    }

    // --- Position and State Accessors ---

    /// Returns the current logical cursor position as `(column, row)`.
    ///
    /// If origin mode is active (as indicated by the `ScreenContext` passed to
    /// movement methods), `row` is relative to the top of the scrolling region.
    /// Otherwise, `row` is relative to the physical top of the screen.
    /// `column` can be equal to `context.width` if the cursor is positioned
    /// after the last character of a line (indicating a wrap on next print).
    #[must_use]
    pub fn logical_pos(&self) -> (usize, usize) {
        (self.cursor.logical_x, self.cursor.logical_y)
    }

    /// Calculates and returns the absolute physical screen position `(column, row)` of the cursor
    /// for rendering or placing a glyph.
    /// This translation considers the current logical position and the provided `ScreenContext`.
    /// The returned coordinates are always absolute to the physical screen grid and are clamped
    /// to be within the screen's dimensions (i.e., `0 <= column < width`).
    ///
    /// # Arguments
    /// * `context`: A reference to the current `ScreenContext`.
    #[must_use]
    pub fn physical_screen_pos(&self, context: &ScreenContext) -> (usize, usize) {
        let physical_y = if context.origin_mode_active {
            let max_relative_y = context.scroll_bot.saturating_sub(context.scroll_top);
            let relative_y_clamped = min(self.cursor.logical_y, max_relative_y);
            context.scroll_top + relative_y_clamped
        } else {
            self.cursor.logical_y
        };

        // For physical positioning (e.g., drawing a glyph or cursor),
        // logical_x == width should be treated as being at the start of the next line (conceptually)
        // or at the last valid cell if we're just getting the cell under the cursor.
        // The actual cell index must be < width.
        let final_x = min(self.cursor.logical_x, context.width.saturating_sub(1));
        let final_y = min(physical_y, context.height.saturating_sub(1));

        (final_x, final_y)
    }

    /// Sets the current SGR attributes for the cursor.
    /// These attributes will be used for subsequently printed characters.
    pub fn set_attributes(&mut self, attributes: Attributes) {
        trace!("Cursor attributes set to: {:?}", attributes);
        self.cursor.attributes = attributes;
    }

    /// Gets a copy of the current SGR attributes of the cursor.
    #[must_use]
    pub fn attributes(&self) -> Attributes {
        self.cursor.attributes
    }

    /// Sets the visibility of the cursor.
    pub fn set_visible(&mut self, visibility: CursorVisibility) {
        trace!("Cursor visibility set to: {:?}", visibility);
        self.cursor.visible = match visibility {
            CursorVisibility::Visible => true,
            CursorVisibility::Hidden => false,
        };
    }

    /// Returns `true` if the cursor is currently set to be visible.
    #[must_use]
    pub fn is_visible(&self) -> bool {
        self.cursor.visible
    }

    // --- Logical Movement Methods ---

    /// Moves the cursor to a new logical position `(new_x, new_y)`.
    ///
    /// The `new_y` coordinate is interpreted relative to the current origin.
    /// The position is clamped to the valid logical area defined by the `ScreenContext`.
    /// For direct positioning like CUP, `new_x` is clamped to `width - 1`.
    ///
    /// # Arguments
    /// * `new_x`: The target logical column. Clamped to `0 <= new_x < width`.
    /// * `new_y`: The target logical row.
    /// * `context`: A reference to the current `ScreenContext` for boundary checking.
    pub fn move_to_logical(&mut self, new_x: usize, new_y: usize, context: &ScreenContext) {
        // For direct positioning (e.g., CUP), logical_x should be within the cell grid (0 to width-1).
        let max_x_for_positioning = context.width.saturating_sub(1);
        let max_y_logical = if context.origin_mode_active {
            context.scroll_bot.saturating_sub(context.scroll_top)
        } else {
            context.height.saturating_sub(1)
        };

        self.cursor.logical_x = min(new_x, max_x_for_positioning);
        self.cursor.logical_y = min(new_y, max_y_logical);
    }

    /// Moves the cursor up by `n` logical rows.
    ///
    /// This movement is purely logical and respects the top boundary of the current
    /// logical screen (or the top of the scrolling region if origin mode is active).
    pub fn move_up(&mut self, n: usize) {
        self.cursor.logical_y = self.cursor.logical_y.saturating_sub(n);
    }

    /// Moves the cursor down by `n` logical rows.
    ///
    /// This movement respects the bottom boundary of the logical screen or scrolling
    /// region as defined by the provided `ScreenContext`.
    ///
    /// # Arguments
    /// * `n`: Number of rows to move down.
    /// * `context`: A reference to the current `ScreenContext`.
    pub fn move_down(&mut self, n: usize, context: &ScreenContext) {
        let max_y_logical = if context.origin_mode_active {
            context.scroll_bot.saturating_sub(context.scroll_top)
        } else {
            context.height.saturating_sub(1)
        };
        self.cursor.logical_y = min(self.cursor.logical_y.saturating_add(n), max_y_logical);
    }

    /// Moves the cursor left by `n` columns.
    ///
    /// This movement is purely logical and respects the left boundary (column 0).
    pub fn move_left(&mut self, n: usize) {
        self.cursor.logical_x = self.cursor.logical_x.saturating_sub(n);
    }

    /// Moves the cursor right by `n` columns, advancing its logical position.
    ///
    /// This movement respects the right boundary of the screen as defined by `context.width`.
    /// `logical_x` can be set to `context.width` to indicate the position *after* the last cell,
    /// signaling that the next character print should wrap.
    ///
    /// # Arguments
    /// * `n`: Number of columns to move right.
    /// * `context`: A reference to the current `ScreenContext`.
    pub fn move_right(&mut self, n: usize, context: &ScreenContext) {
        // Allow logical_x to reach context.width for wrap signaling.
        let max_x_for_advancing = context.width;
        self.cursor.logical_x = min(self.cursor.logical_x.saturating_add(n), max_x_for_advancing);
    }

    /// Moves the cursor to a specific logical column `new_x`.
    ///
    /// This movement respects the right boundary of the screen as defined by `context.width`.
    /// `new_x` is clamped to `width - 1` for direct positioning.
    /// The logical row remains unchanged.
    ///
    /// # Arguments
    /// * `new_x`: The target logical column. Clamped to `0 <= new_x < width`.
    /// * `context`: A reference to the current `ScreenContext`.
    pub fn move_to_logical_col(&mut self, new_x: usize, context: &ScreenContext) {
        let max_x_for_positioning = context.width.saturating_sub(1);
        self.cursor.logical_x = min(new_x, max_x_for_positioning);
    }

    /// Moves the cursor to a specific logical row `new_y` (VPA), leaving the
    /// logical column unchanged. `new_y` is clamped exactly as `move_to_logical`
    /// clamps its row argument (bottom of the scrolling region under origin
    /// mode, otherwise the bottom of the screen).
    ///
    /// # Arguments
    /// * `new_y`: The target logical row.
    /// * `context`: A reference to the current `ScreenContext` for boundary checking.
    pub fn move_to_logical_row(&mut self, new_y: usize, context: &ScreenContext) {
        let max_y_logical = if context.origin_mode_active {
            context.scroll_bot.saturating_sub(context.scroll_top)
        } else {
            context.height.saturating_sub(1)
        };
        self.cursor.logical_y = min(new_y, max_y_logical);
    }

    /// Moves the cursor to the start of the current logical line (column 0).
    /// The logical row remains unchanged.
    pub fn carriage_return(&mut self) {
        self.cursor.logical_x = 0;
    }

    // --- Save/Restore Functionality (DECSC/DECRC) ---

    /// Saves the current cursor state (logical position, attributes, visibility).
    /// This corresponds to the DECSC (Save Cursor) sequence.
    pub fn save_state(&mut self) {
        self.saved_cursor = Some(self.cursor);
        trace!("Cursor state saved: {:?}", self.cursor);
    }

    /// Restores the cursor state from the last save.
    /// This corresponds to the DECRC (Restore Cursor) sequence.
    ///
    /// If no state was previously saved, the cursor is reset to a default state:
    /// logical position (0,0), `default_attributes`, and visible.
    /// The restored logical position is clamped to the current valid boundaries
    /// defined by the `ScreenContext` (logical_x clamped to `width - 1`).
    ///
    /// # Arguments
    /// * `context`: A reference to the current `ScreenContext` for boundary clamping.
    /// * `default_attributes`: The `Attributes` to use if no saved state exists.
    pub fn restore_state(&mut self, context: &ScreenContext, default_attributes: Attributes) {
        let state_to_restore = self.saved_cursor.unwrap_or_else(|| {
            trace!("No saved cursor state, restoring to default logical (0,0).");
            Cursor {
                logical_x: 0,
                logical_y: 0,
                attributes: default_attributes,
                visible: true, // Default visibility on restore if nothing was saved.
                shape: CursorShape::default(),
                unfocused_shape: CursorShape::unfocused_default(),
            }
        });

        self.cursor = state_to_restore;

        // Re-clamp the restored logical position to current boundaries.
        // For direct positioning, logical_x should be within the cell grid (0 to width-1).
        let max_x_for_positioning = context.width.saturating_sub(1);
        let max_y_logical = if context.origin_mode_active {
            context.scroll_bot.saturating_sub(context.scroll_top)
        } else {
            context.height.saturating_sub(1)
        };
        self.cursor.logical_x = min(self.cursor.logical_x, max_x_for_positioning);
        self.cursor.logical_y = min(self.cursor.logical_y, max_y_logical);
        trace!(
            "Cursor state restored to: {:?}. Context: {:?}",
            self.cursor,
            context
        );
    }
    pub fn reset(&mut self) {
        self.set_attributes(Attributes::default());
        self.set_visible(CursorVisibility::Visible);
        self.cursor = Cursor::default();
    }
}

// Serde Serialize + Deserialize Cursor Shape as string rather than u16
impl Serialize for CursorShape {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.to_str())
    }
}

struct CursorShapeVisitor;

impl<'de> Visitor<'de> for CursorShapeVisitor {
    type Value = CursorShape;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter
            .write_str("a string representing a CursorShape (e.g., 'BlinkingBlock', 'SteadyBlock')")
    }

    fn visit_str<E>(self, value: &str) -> Result<CursorShape, E>
    where
        E: de::Error,
    {
        CursorShape::from_str(value).map_err(|_| {
            de::Error::unknown_variant(
                value,
                &[
                    "BlinkingBlock",
                    "SteadyBlock",
                    "BlinkingUnderline",
                    "SteadyUnderline",
                    "BlinkingBar",
                    "SteadyBar",
                ],
            )
        })
    }
}

impl<'de> Deserialize<'de> for CursorShape {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(CursorShapeVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::{Cursor, CursorController, CursorShape};
    use crate::{glyph::Attributes, term::cursor_visibility::CursorVisibility};

    #[test]
    fn it_should_map_each_documented_decscusr_code_to_its_own_distinct_shape() {
        let shapes: Vec<CursorShape> = (1..=6).map(CursorShape::from_decscusr_code).collect();
        assert_eq!(
            shapes,
            vec![
                CursorShape::BlinkingBlock,
                CursorShape::SteadyBlock,
                CursorShape::BlinkingUnderline,
                CursorShape::SteadyUnderline,
                CursorShape::BlinkingBar,
                CursorShape::SteadyBar,
            ],
            "each DECSCUSR code 1-6 must map to a unique CursorShape variant"
        );
    }

    #[test]
    fn it_should_map_decscusr_code_zero_to_the_default_shape() {
        assert_eq!(CursorShape::from_decscusr_code(0), CursorShape::default());
    }

    #[test]
    fn it_should_map_an_unrecognized_decscusr_code_to_the_default_shape() {
        assert_eq!(CursorShape::from_decscusr_code(99), CursorShape::default());
    }

    #[test]
    fn it_should_round_trip_every_shape_through_serde_json() {
        let shapes = [
            CursorShape::BlinkingBlock,
            CursorShape::SteadyBlock,
            CursorShape::BlinkingUnderline,
            CursorShape::SteadyUnderline,
            CursorShape::BlinkingBar,
            CursorShape::SteadyBar,
        ];
        for shape in shapes {
            let json = serde_json::to_string(&shape).expect("serialize");
            let round_tripped: CursorShape = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(
                round_tripped, shape,
                "round trip through {json} changed the shape"
            );
        }
    }

    #[test]
    fn it_should_serialize_each_shape_to_its_own_distinct_json_string() {
        let shapes = [
            CursorShape::BlinkingBlock,
            CursorShape::SteadyBlock,
            CursorShape::BlinkingUnderline,
            CursorShape::SteadyUnderline,
            CursorShape::BlinkingBar,
            CursorShape::SteadyBar,
        ];
        let jsons: Vec<String> = shapes
            .iter()
            .map(|s| serde_json::to_string(s).expect("serialize"))
            .collect();
        let mut unique = jsons.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(
            unique.len(),
            jsons.len(),
            "expected each CursorShape to serialize to a distinct string, got {jsons:?}"
        );
    }

    #[test]
    fn it_should_reject_an_unknown_shape_name_when_deserializing() {
        let result: Result<CursorShape, _> = serde_json::from_str("\"NotAShape\"");
        assert!(result.is_err());
    }

    #[test]
    fn it_should_describe_the_expected_shape_string_when_deserializing_the_wrong_json_type() {
        let result: Result<CursorShape, _> = serde_json::from_str("42");
        let err = result.expect_err("a JSON number is not a valid CursorShape encoding");
        assert!(
            err.to_string()
                .contains("a string representing a CursorShape"),
            "deserialize error should describe the expected value, got: {err}"
        );
    }

    #[test]
    fn it_should_default_the_unfocused_shape_independently_from_the_focused_shape() {
        let cursor = Cursor::default();
        // The default config gives focused and unfocused cursors distinct shapes
        // (SteadyBlock vs SteadyBar) -- unfocused_shape must not just mirror shape.
        assert_ne!(cursor.shape, cursor.unfocused_shape);
    }

    #[test]
    fn it_should_start_visible_and_toggle_visibility_via_set_visible() {
        let mut controller = CursorController::new(Attributes::default());
        assert!(controller.is_visible());

        controller.set_visible(CursorVisibility::Hidden);
        assert!(!controller.is_visible());

        controller.set_visible(CursorVisibility::Visible);
        assert!(controller.is_visible());
    }

    #[test]
    fn it_should_restore_default_attributes_position_and_visibility_on_reset() {
        let custom_attributes = Attributes {
            fg: crate::color::Color::Indexed(9),
            ..Attributes::default()
        };
        let context = super::ScreenContext {
            width: 80,
            height: 24,
            scroll_top: 0,
            scroll_bot: 23,
            origin_mode_active: false,
        };
        let mut controller = CursorController::new(custom_attributes);
        controller.set_visible(CursorVisibility::Hidden);
        controller.move_to_logical(10, 5, &context);
        assert_eq!(controller.logical_pos(), (10, 5));

        controller.reset();

        assert!(controller.is_visible());
        assert_eq!(controller.attributes(), Attributes::default());
        assert_eq!(controller.logical_pos(), (0, 0));
    }
}
