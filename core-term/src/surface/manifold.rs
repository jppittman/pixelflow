//! Terminal Manifold - A Conal Elliott style terminal renderer.
//!
//! The terminal is expressed as a composition of manifold combinators.
//! The type IS the AST - the grid is a tree of Select combinators that
//! the compiler monomorphizes to efficient SIMD code.
//!
//! # Architecture
//!
//! A terminal grid is built as a binary search tree of `Select` nodes:
//!
//! ```text
//! color_manifold(
//!   Select { cond: Lt(X, mid), if_true: left_r, if_false: right_r },
//!   Select { cond: Lt(X, mid), if_true: left_g, if_false: right_g },
//!   ...
//! )
//! ```
//!
//! This gives O(log(cols) + log(rows)) depth, enabling fully vectorized
//! evaluation without extracting scalar values from SIMD lanes.
//!
//! # Key Insight
//!
//! A manifold is a functor. We never extract values - we compose manifolds.
//! The grid lookup is itself expressed as manifold composition.
//!
//! Uses `ColorManifold` from pixelflow-graphics for RGBA packing.
//! Uses `Color` from pixelflow-graphics for solid color manifolds.

use pixelflow_core::{
    Field, Lt, Manifold, ManifoldCompat, ManifoldExpr, ManifoldExt, Select, X, Y,
};

/// The standard 4D Field domain type.
type Field4 = (Field, Field, Field, Field);
use pixelflow_graphics::render::color::{color_manifold, ColorManifold};

// Re-export Color for solid color manifolds
pub use pixelflow_graphics::render::color::Color;

// ============================================================================
// Cell: Glyph Coverage + Colors → RGBA
// ============================================================================

/// A terminal cell that blends foreground/background based on glyph coverage.
///
/// This is the leaf node in the terminal manifold tree. It takes a glyph
/// (coverage manifold outputting Field) and produces a color blend.
#[derive(Clone)]
pub struct Cell<G> {
    /// The glyph coverage manifold (0.0 = background, 1.0 = foreground).
    pub glyph: G,
    /// Foreground color (R, G, B, A) normalized to [0, 1].
    pub fg: [f32; 4],
    /// Background color (R, G, B, A) normalized to [0, 1].
    pub bg: [f32; 4],
}

impl<G: Manifold<Output = Field>> Cell<G> {
    /// Create a new cell with glyph and colors.
    pub fn new(glyph: G, fg: [f32; 4], bg: [f32; 4]) -> Self {
        Self { glyph, fg, bg }
    }
}

/// Extracts a single channel from a Cell as a Field manifold.
///
/// This enables using the standard Select combinator (which works on Field)
/// for grid lookup.
#[derive(Clone)]
pub struct CellChannel<G, const CHANNEL: usize> {
    cell: Cell<G>,
}

impl<G, const CHANNEL: usize> ManifoldExpr for CellChannel<G, CHANNEL> {}

impl<G, const CHANNEL: usize> CellChannel<G, CHANNEL> {
    /// Create a channel extractor for a cell.
    pub fn new(cell: Cell<G>) -> Self {
        Self { cell }
    }
}

impl<G: ManifoldCompat<Field, Output = Field> + Clone, const CHANNEL: usize> Manifold<Field4>
    for CellChannel<G, CHANNEL>
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        let (x, y, z, w) = p;
        let zero = Field::from(0.0);
        let coverage = self.cell.glyph.eval_raw(x, y, z, w);
        let c = coverage.max(zero).min(Field::from(1.0)).constant();
        let omc = Field::from(1.0) - c;
        (c * Field::from(self.cell.fg[CHANNEL]) + omc * Field::from(self.cell.bg[CHANNEL]))
            .constant()
    }
}

// Type aliases for each channel
pub type CellR<G> = CellChannel<G, 0>;
pub type CellG<G> = CellChannel<G, 1>;
pub type CellB<G> = CellChannel<G, 2>;
pub type CellA<G> = CellChannel<G, 3>;

// ============================================================================
// Local Coordinate Transform
// ============================================================================

/// Transforms coordinates to be local within a cell.
#[derive(Clone, Copy, Debug)]
pub struct LocalCoords<M> {
    pub inner: M,
    pub offset_x: f32,
    pub offset_y: f32,
}

impl<M> LocalCoords<M> {
    pub fn new(inner: M, offset_x: f32, offset_y: f32) -> Self {
        Self {
            inner,
            offset_x,
            offset_y,
        }
    }
}

impl<M: ManifoldCompat<Field, Output = Field> + ManifoldExt> Manifold<Field4> for LocalCoords<M> {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        let (x, y, z, w) = p;
        let local_x = x - Field::from(self.offset_x);
        let local_y = y - Field::from(self.offset_y);
        self.inner.eval_at(local_x, local_y, z, w)
    }
}

impl<M> ManifoldExpr for LocalCoords<M> {}

// ============================================================================
// Grid Builder Using ColorManifold + Select
// ============================================================================

/// Trait for types that can provide cell manifolds.
pub trait CellFactory: Send + Sync {
    /// The glyph manifold type.
    type Glyph: Manifold<Output = Field> + Clone + Send + Sync + 'static;

    /// Create a cell's glyph manifold for the given grid position.
    fn glyph(&self, col: usize, row: usize) -> Self::Glyph;

    /// Get foreground color for a cell.
    fn fg(&self, col: usize, row: usize) -> [f32; 4];

    /// Get background color for a cell.
    fn bg(&self, col: usize, row: usize) -> [f32; 4];

    /// Grid dimensions (cols, rows).
    fn dimensions(&self) -> (usize, usize);

    /// Cell dimensions in pixels.
    fn cell_size(&self) -> (f32, f32);
}

/// Builds a terminal grid as a ColorManifold with Select trees per channel.
///
/// The result is a `ColorManifold` where each channel (R, G, B, A) is a
/// binary search tree of Select combinators.
///
/// Type complexity is O(cols * rows) for the full grid structure.
pub fn build_grid<F: CellFactory>(
    factory: &F,
) -> ColorManifold<
    impl Manifold<Output = Field>,
    impl Manifold<Output = Field>,
    impl Manifold<Output = Field>,
    impl Manifold<Output = Field>,
> {
    let (cols, rows) = factory.dimensions();
    let (cell_w, cell_h) = factory.cell_size();

    // Build a Select tree for each channel
    let r = build_channel_tree::<F, 0>(factory, 0, cols, 0, rows, cell_w, cell_h);
    let g = build_channel_tree::<F, 1>(factory, 0, cols, 0, rows, cell_w, cell_h);
    let b = build_channel_tree::<F, 2>(factory, 0, cols, 0, rows, cell_w, cell_h);
    let a = build_channel_tree::<F, 3>(factory, 0, cols, 0, rows, cell_w, cell_h);

    color_manifold(r, g, b, a)
}

/// Build a Select tree for a single color channel.
#[allow(clippy::too_many_arguments)]
fn build_channel_tree<F: CellFactory, const CHANNEL: usize>(
    factory: &F,
    col_start: usize,
    col_end: usize,
    row_start: usize,
    row_end: usize,
    cell_w: f32,
    cell_h: f32,
) -> Box<dyn Manifold<Output = Field> + Send + Sync> {
    let col_count = col_end - col_start;
    let row_count = row_end - row_start;

    // Base case: single cell
    if col_count == 1 && row_count == 1 {
        let glyph = factory.glyph(col_start, row_start);
        let fg = factory.fg(col_start, row_start);
        let bg = factory.bg(col_start, row_start);

        let cell = Cell::new(glyph, fg, bg);
        let x_offset = col_start as f32 * cell_w;
        let y_offset = row_start as f32 * cell_h;

        // Extract the channel with local coordinate transform
        return Box::new(LocalCoords::new(
            CellChannel::<_, CHANNEL>::new(cell),
            x_offset,
            y_offset,
        ));
    }

    // Split on the larger dimension
    if col_count >= row_count && col_count > 1 {
        // Split columns
        let mid = col_start + col_count / 2;
        let threshold = mid as f32 * cell_w;

        let left = build_channel_tree::<F, CHANNEL>(
            factory, col_start, mid, row_start, row_end, cell_w, cell_h,
        );
        let right = build_channel_tree::<F, CHANNEL>(
            factory, mid, col_end, row_start, row_end, cell_w, cell_h,
        );

        // X < threshold ? left : right
        Box::new(Select {
            cond: Lt(X, threshold),
            if_true: left,
            if_false: right,
        })
    } else {
        // Split rows
        let mid = row_start + row_count / 2;
        let threshold = mid as f32 * cell_h;

        let top = build_channel_tree::<F, CHANNEL>(
            factory, col_start, col_end, row_start, mid, cell_w, cell_h,
        );
        let bottom = build_channel_tree::<F, CHANNEL>(
            factory, col_start, col_end, mid, row_end, cell_w, cell_h,
        );

        // Y < threshold ? top : bottom
        Box::new(Select {
            cond: Lt(Y, threshold),
            if_true: top,
            if_false: bottom,
        })
    }
}

// ============================================================================
// Constant Coverage (for empty cells or solid blocks)
// ============================================================================

/// A constant coverage manifold.
#[derive(Clone, Copy, Debug)]
pub struct ConstCoverage(pub f32);

impl Manifold<Field4> for ConstCoverage {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, _p: Field4) -> Field {
        Field::from(self.0)
    }
}

impl ManifoldExpr for ConstCoverage {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    struct MockFactory {
        cols: usize,
        rows: usize,
        cell_w: f32,
        cell_h: f32,
    }

    impl CellFactory for MockFactory {
        type Glyph = ConstCoverage;

        fn glyph(&self, _col: usize, _row: usize) -> Self::Glyph {
            ConstCoverage(1.0) // Full coverage
        }

        fn fg(&self, col: usize, row: usize) -> [f32; 4] {
            // Alternate white/black based on position
            if (col + row).is_multiple_of(2) {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.0, 0.0, 0.0, 1.0]
            }
        }

        fn bg(&self, _col: usize, _row: usize) -> [f32; 4] {
            [0.5, 0.5, 0.5, 1.0]
        }

        fn dimensions(&self) -> (usize, usize) {
            (self.cols, self.rows)
        }

        fn cell_size(&self) -> (f32, f32) {
            (self.cell_w, self.cell_h)
        }
    }

    #[test]
    fn grid_construction_should_succeed_when_invoked() {
        use pixelflow_graphics::render::rasterizer::rasterize;

        let factory = MockFactory {
            cols: 4,
            rows: 4,
            cell_w: 8.0,
            cell_h: 16.0,
        };

        let grid = build_grid(&factory);

        // Render a small region to test
        use pixelflow_graphics::render::color::Bgra8;
        use pixelflow_graphics::render::frame::Frame;
        let mut frame = Frame::<Bgra8>::new(8, 8);
        rasterize(&grid, &mut frame, 1);

        // Check pixel at (4, 0) - should be in first cell (0,0) - white
        let (row, col) = (0usize, 4usize);
        let pixel_index = row * 8 + col;
        let r = frame.data[pixel_index].r();
        assert!(r > 200, "Expected white, got r={}", r);
    }

    #[test]
    fn cell_channel_blending_should_succeed_when_invoked() {
        use pixelflow_graphics::render::rasterizer::rasterize;

        let cell = Cell::new(
            ConstCoverage(0.5),
            [1.0, 0.0, 0.0, 1.0], // Red
            [0.0, 0.0, 1.0, 1.0], // Blue
        );

        // Use ColorManifold to pack channels
        let packed = color_manifold(
            CellR::new(cell.clone()),
            CellG::new(cell.clone()),
            CellB::new(cell.clone()),
            CellA::new(cell),
        );

        // Render a single pixel
        use pixelflow_graphics::render::color::Bgra8;
        use pixelflow_graphics::render::frame::Frame;
        let mut frame = Frame::<Bgra8>::new(1, 1);
        rasterize(&packed, &mut frame, 1);

        // 50% coverage: R = 0.5*1.0 + 0.5*0.0 = 0.5 = 127
        // B = 0.5*0.0 + 0.5*1.0 = 0.5 = 127
        let r = frame.data[0].r();
        let b = frame.data[0].b();

        assert!(r > 100 && r < 160, "Expected ~127, got r={}", r);
        assert!(b > 100 && b < 160, "Expected ~127, got b={}", b);
    }

    #[test]
    fn color_manifold_should_succeed_when_invoked() {
        use pixelflow_graphics::render::rasterizer::rasterize;

        // Color::Rgb from pixelflow-graphics implements Manifold<Output = Discrete>
        let red = Color::Rgb(255, 0, 0);

        // Render a single pixel
        use pixelflow_graphics::render::color::Rgba8;
        use pixelflow_graphics::render::frame::Frame;
        let mut frame = Frame::<Rgba8>::new(1, 1);
        rasterize(&red, &mut frame, 1);

        let pixel = frame.data[0];
        // Check that red channel is approximately correct (might have some rounding)
        assert!(
            pixel.r() > 200,
            "Red channel should be > 200, got {}",
            pixel.r()
        );
        assert_eq!(pixel.g(), 0);
        assert_eq!(pixel.b(), 0);
    }
}
