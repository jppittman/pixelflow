// src/term/layout.rs
//
// Layout Manager - The "Unrenderer"
//
// This module handles the geometric mapping between physical pixels and terminal grid cells.
// It encapsulates all coordinate transformation logic, making it the single source of truth
// for "where is (x, y)?" questions.

use crate::config::CONFIG;

/// Manages the geometric layout of the terminal grid.
///
/// The Layout is responsible for:
/// - Tracking grid dimensions (cols, rows)
/// - Tracking cell dimensions (width, height in pixels)
/// - Converting between pixel coordinates and cell coordinates ("unrendering")
/// - Handling padding/borders
///
/// This struct is owned by the TerminalEmulator and used for all spatial calculations.
#[derive(Debug, Clone)]
pub struct Layout {
    /// Number of columns in the terminal grid
    pub cols: usize,

    /// Number of rows in the terminal grid
    pub rows: usize,

    /// Width of a single cell in pixels
    pub cell_width_px: usize,

    /// Height of a single cell in pixels
    pub cell_height_px: usize,

    /// Horizontal padding/border in pixels (future use)
    pub padding_x: u16,

    /// Vertical padding/border in pixels (future use)
    pub padding_y: u16,
}

impl Layout {
    /// Creates a new Layout with the specified grid dimensions.
    ///
    /// Cell dimensions are read from CONFIG.appearance.
    #[must_use]
    pub fn new(cols: usize, rows: usize) -> Self {
        Self {
            cols,
            rows,
            cell_width_px: CONFIG.appearance.cell_width_px,
            cell_height_px: CONFIG.appearance.cell_height_px,
            padding_x: 0, // No padding yet, but ready for future use
            padding_y: 0,
        }
    }

    /// The core "unrender" function.
    ///
    /// Converts physical pixel coordinates (from mouse events) to logical grid coordinates.
    ///
    /// Returns None if the coordinates are outside the terminal grid (in padding or beyond).
    ///
    /// # Arguments
    /// * `x_px` - X coordinate in pixels (logical points, not physical pixels)
    /// * `y_px` - Y coordinate in pixels (logical points, not physical pixels)
    ///
    /// # Returns
    /// * `Some((col, row))` if the coordinates are within the grid
    /// * `None` if the coordinates are in padding or outside the grid
    #[must_use]
    pub fn pixels_to_cells(&self, x_px: u16, y_px: u16) -> Option<(usize, usize)> {
        // Apply padding (if any)
        if x_px < self.padding_x || y_px < self.padding_y {
            return None; // Click in the border/margin
        }

        let effective_x = x_px - self.padding_x;
        let effective_y = y_px - self.padding_y;

        // Convert to cell coordinates
        let col = (effective_x as usize) / self.cell_width_px.max(1);
        let row = (effective_y as usize) / self.cell_height_px.max(1);

        // Bounds check
        if col >= self.cols || row >= self.rows {
            return None; // Click outside the grid (right/bottom padding or beyond)
        }

        Some((col, row))
    }

    /// Updates the grid dimensions.
    ///
    /// Called when the terminal is resized.
    pub fn resize(&mut self, cols: usize, rows: usize) {
        self.cols = cols;
        self.rows = rows;
    }

    /// Calculates the total window pixel dimensions required for this layout.
    ///
    /// Includes padding on all sides.
    ///
    /// # Returns
    /// `(width_px, height_px)` tuple
    #[must_use]
    pub fn pixel_dimensions(&self) -> (u32, u32) {
        let w = (self.cols * self.cell_width_px) + (self.padding_x as usize * 2);
        let h = (self.rows * self.cell_height_px) + (self.padding_y as usize * 2);
        (w as u32, h as u32)
    }

    /// Converts cell coordinates to pixel coordinates (top-left corner of the cell).
    ///
    /// Useful for cursor positioning and rendering.
    ///
    /// # Returns
    /// `(x_px, y_px)` tuple representing the top-left corner of the cell
    #[must_use]
    pub fn cells_to_pixels(&self, col: usize, row: usize) -> (u16, u16) {
        let x = self.padding_x + (col * self.cell_width_px) as u16;
        let y = self.padding_y + (row * self.cell_height_px) as u16;
        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixels_to_cells_basic() {
        let layout = Layout {
            cols: 80,
            rows: 24,
            cell_width_px: 10,
            cell_height_px: 20,
            padding_x: 0,
            padding_y: 0,
        };

        // Top-left cell
        assert_eq!(layout.pixels_to_cells(0, 0), Some((0, 0)));
        assert_eq!(layout.pixels_to_cells(5, 10), Some((0, 0)));

        // Cell (1, 0)
        assert_eq!(layout.pixels_to_cells(10, 0), Some((1, 0)));

        // Cell (0, 1)
        assert_eq!(layout.pixels_to_cells(0, 20), Some((0, 1)));

        // Middle cell
        assert_eq!(layout.pixels_to_cells(395, 239), Some((39, 11)));
    }

    #[test]
    fn test_pixels_to_cells_with_padding() {
        let layout = Layout {
            cols: 80,
            rows: 24,
            cell_width_px: 10,
            cell_height_px: 20,
            padding_x: 5,
            padding_y: 10,
        };

        // Click in padding
        assert_eq!(layout.pixels_to_cells(0, 0), None);
        assert_eq!(layout.pixels_to_cells(4, 9), None);

        // Top-left cell (accounting for padding)
        assert_eq!(layout.pixels_to_cells(5, 10), Some((0, 0)));

        // Cell (1, 0)
        assert_eq!(layout.pixels_to_cells(15, 10), Some((1, 0)));
    }

    #[test]
    fn test_pixels_to_cells_out_of_bounds() {
        let layout = Layout {
            cols: 80,
            rows: 24,
            cell_width_px: 10,
            cell_height_px: 20,
            padding_x: 0,
            padding_y: 0,
        };

        // Beyond right edge (col 80 doesn't exist)
        assert_eq!(layout.pixels_to_cells(800, 0), None);

        // Beyond bottom edge (row 24 doesn't exist)
        assert_eq!(layout.pixels_to_cells(0, 480), None);

        // Way out of bounds
        assert_eq!(layout.pixels_to_cells(10000, 10000), None);
    }

    #[test]
    fn test_cells_to_pixels() {
        let layout = Layout {
            cols: 80,
            rows: 24,
            cell_width_px: 10,
            cell_height_px: 20,
            padding_x: 0,
            padding_y: 0,
        };

        assert_eq!(layout.cells_to_pixels(0, 0), (0, 0));
        assert_eq!(layout.cells_to_pixels(1, 0), (10, 0));
        assert_eq!(layout.cells_to_pixels(0, 1), (0, 20));
        assert_eq!(layout.cells_to_pixels(79, 23), (790, 460));
    }

    #[test]
    fn test_pixel_dimensions() {
        let layout = Layout {
            cols: 80,
            rows: 24,
            cell_width_px: 10,
            cell_height_px: 20,
            padding_x: 5,
            padding_y: 10,
        };

        // 80 * 10 + 5 * 2 = 810
        // 24 * 20 + 10 * 2 = 500
        assert_eq!(layout.pixel_dimensions(), (810, 500));
    }

    #[test]
    fn test_resize() {
        let mut layout = Layout::new(80, 24);

        assert_eq!(layout.cols, 80);
        assert_eq!(layout.rows, 24);

        layout.resize(100, 30);

        assert_eq!(layout.cols, 100);
        assert_eq!(layout.rows, 30);
    }
}
