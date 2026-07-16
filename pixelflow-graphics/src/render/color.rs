// pixelflow-render/src/color.rs
//! Unified color types for terminal rendering.
//!
//! This module provides:
//! - **Semantic colors**: `Color` enum for high-level specification
//! - **Pixel formats**: `Rgba8`, `Bgra8` for framebuffer storage
//! - **Color manifolds**: `ColorManifold`, `Grayscale`, `ColorCube` for functional color composition
//!
//! For color manifolds, use `pixelflow_core::{Rgba, Red, Green, Blue, Alpha}`.

use bitflags::bitflags;
use pixelflow_core::{At, Discrete, Field, Manifold, ManifoldCompat};

/// The standard 4D Field domain type.
type Field4 = (Field, Field, Field, Field);

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// Re-export the Pixel trait from the local pixel module
pub use super::pixel::Pixel;

// =============================================================================
// Semantic Color Types (The "User Input" tier)
// =============================================================================

/// Standard ANSI named colors (indices 0-15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u8)]
pub enum NamedColor {
    /// ANSI Black.
    Black = 0,
    /// ANSI Red.
    Red = 1,
    /// ANSI Green.
    Green = 2,
    /// ANSI Yellow.
    Yellow = 3,
    /// ANSI Blue.
    Blue = 4,
    /// ANSI Magenta.
    Magenta = 5,
    /// ANSI Cyan.
    Cyan = 6,
    /// ANSI White.
    White = 7,
    /// ANSI Bright Black.
    BrightBlack = 8,
    /// ANSI Bright Red.
    BrightRed = 9,
    /// ANSI Bright Green.
    BrightGreen = 10,
    /// ANSI Bright Yellow.
    BrightYellow = 11,
    /// ANSI Bright Blue.
    BrightBlue = 12,
    /// ANSI Bright Magenta.
    BrightMagenta = 13,
    /// ANSI Bright Cyan.
    BrightCyan = 14,
    /// ANSI Bright White.
    BrightWhite = 15,
}

impl NamedColor {
    /// Convert a u8 index (0-15) to a NamedColor.
    pub fn from_index(idx: u8) -> Self {
        assert!(idx < 16, "Invalid NamedColor index: {}. Must be 0-15.", idx);
        unsafe { core::mem::transmute(idx) }
    }

    /// Returns the RGB representation of this named color.
    pub fn to_rgb(self) -> (u8, u8, u8) {
        ANSI_COLORS_RGB[self as usize]
    }
}

// Make NamedColor a manifold - an infinite field of that ANSI color
impl pixelflow_core::Manifold<Field4> for NamedColor {
    type Output = pixelflow_core::Discrete;

    #[inline(always)]
    fn eval(&self, p: Field4) -> pixelflow_core::Discrete {
        let (x, y, z, w) = p;
        let (r, g, b) = self.to_rgb();
        // Use RGBA ColorCube terminal object directly
        RgbaColorCube::default()
            .at(
                Field::from(r as f32 / 255.0),
                Field::from(g as f32 / 255.0),
                Field::from(b as f32 / 255.0),
                Field::from(1.0),
            )
            .eval_raw(x, y, z, w)
    }
}

/// Represents a semantic color value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Color {
    /// Default foreground or background color.
    #[default]
    Default,
    /// A standard named ANSI color (indices 0-15).
    Named(NamedColor),
    /// An indexed color from the 256-color palette (indices 0-255).
    Indexed(u8),
    /// An RGB true color.
    Rgb(u8, u8, u8),
}

impl Color {
    /// Convert to an Rgba8 pixel.
    #[inline]
    pub fn to_rgba8(self) -> Rgba8 {
        Rgba8(u32::from(self))
    }

    /// Convert to a Bgra8 pixel.
    #[inline]
    pub fn to_bgra8(self) -> Bgra8 {
        Bgra8::from(self.to_rgba8())
    }

    /// Convert to normalized f32 RGBA components.
    #[inline]
    pub fn to_f32_rgba(self) -> (f32, f32, f32, f32) {
        let rgba = self.to_rgba8();
        (
            rgba.r() as f32 / 255.0,
            rgba.g() as f32 / 255.0,
            rgba.b() as f32 / 255.0,
            rgba.a() as f32 / 255.0,
        )
    }
}

// Make Color a manifold - an infinite field of that color
impl pixelflow_core::Manifold<Field4> for Color {
    type Output = pixelflow_core::Discrete;

    #[inline(always)]
    fn eval(&self, p: Field4) -> pixelflow_core::Discrete {
        let (x, y, z, w) = p;
        let (r, g, b, a) = self.to_f32_rgba();
        RgbaColorCube::default()
            .at(
                Field::from(r),
                Field::from(g),
                Field::from(b),
                Field::from(a),
            )
            .eval_raw(x, y, z, w)
    }
}

// Constants for 256-color palette conversion
const ANSI_NAMED_COLOR_COUNT: u8 = 16;
const COLOR_CUBE_OFFSET: u8 = 16;
const COLOR_CUBE_SIZE: u8 = 6;
const COLOR_CUBE_TOTAL_COLORS: u8 = COLOR_CUBE_SIZE * COLOR_CUBE_SIZE * COLOR_CUBE_SIZE;
const GRAYSCALE_OFFSET: u8 = COLOR_CUBE_OFFSET + COLOR_CUBE_TOTAL_COLORS;

const CUBE_SCALE_FACTOR: u8 = 40;
const CUBE_BASE_OFFSET: u8 = 55;
const GRAYSCALE_STEP: u8 = 10;
const GRAYSCALE_BASE: u8 = 8;

const ANSI_COLORS_RGB: [(u8, u8, u8); 16] = [
    (0, 0, 0),       // Black
    (205, 0, 0),     // Red
    (0, 205, 0),     // Green
    (205, 205, 0),   // Yellow
    (0, 0, 238),     // Blue
    (205, 0, 205),   // Magenta
    (0, 205, 205),   // Cyan
    (229, 229, 229), // White
    (127, 127, 127), // BrightBlack
    (255, 0, 0),     // BrightRed
    (0, 255, 0),     // BrightGreen
    (255, 255, 0),   // BrightYellow
    (92, 92, 255),   // BrightBlue
    (255, 0, 255),   // BrightMagenta
    (0, 255, 255),   // BrightCyan
    (255, 255, 255), // BrightWhite
];

/// Precomputed 256-color palette lookup table.
/// Stores packed RGBA (0xAABBGGRR) values for O(1) conversion.
static PALETTE: [u32; 256] = generate_palette();

/// Generates the 256-color palette at compile/link time.
const fn generate_palette() -> [u32; 256] {
    let mut palette = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let idx = i as u8;
        let (r, g, b) = if idx < ANSI_NAMED_COLOR_COUNT {
            // Named colors
            ANSI_COLORS_RGB[idx as usize]
        } else if idx < GRAYSCALE_OFFSET {
            // 6x6x6 Color Cube (indices 16-231)
            let cube_idx = idx - COLOR_CUBE_OFFSET;
            let r_comp = (cube_idx / (COLOR_CUBE_SIZE * COLOR_CUBE_SIZE)) % COLOR_CUBE_SIZE;
            let g_comp = (cube_idx / COLOR_CUBE_SIZE) % COLOR_CUBE_SIZE;
            let b_comp = cube_idx % COLOR_CUBE_SIZE;
            let r_val = if r_comp == 0 { 0 } else { r_comp * CUBE_SCALE_FACTOR + CUBE_BASE_OFFSET };
            let g_val = if g_comp == 0 { 0 } else { g_comp * CUBE_SCALE_FACTOR + CUBE_BASE_OFFSET };
            let b_val = if b_comp == 0 { 0 } else { b_comp * CUBE_SCALE_FACTOR + CUBE_BASE_OFFSET };
            (r_val, g_val, b_val)
        } else {
            // Grayscale ramp (indices 232-255)
            let gray_idx = idx - GRAYSCALE_OFFSET;
            let level = gray_idx * GRAYSCALE_STEP + GRAYSCALE_BASE;
            (level, level, level)
        };

        // Pack into u32 (RGBA little-endian: 0xAABBGGRR)
        // u32::from_le_bytes is const-stable since 1.32
        palette[i] = u32::from_le_bytes([r, g, b, 255]);
        i += 1;
    }
    palette
}

impl From<Color> for u32 {
    /// Convert a Color to a u32 pixel value (RGBA format: 0xAABBGGRR).
    #[inline] // Hot path!
    fn from(color: Color) -> u32 {
        match color {
            Color::Default => u32::from_le_bytes([0, 0, 0, 255]),

            // Optimized lookup for Named and Indexed
            Color::Named(named) => PALETTE[named as usize],
            Color::Indexed(idx) => PALETTE[idx as usize],

            // Fallback for TrueColor
            Color::Rgb(r, g, b) => u32::from_le_bytes([r, g, b, 255]),
        }
    }
}

// =============================================================================
// Text Attributes
// =============================================================================

bitflags! {
    /// Text attribute flags (bold, underline, etc.).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    pub struct AttrFlags: u16 {
        const BOLD          = 1 << 0;
        const FAINT         = 1 << 1;
        const ITALIC        = 1 << 2;
        const UNDERLINE     = 1 << 3;
        const BLINK         = 1 << 4;
        const REVERSE       = 1 << 5;
        const HIDDEN        = 1 << 6;
        const STRIKETHROUGH = 1 << 7;
    }
}

// =============================================================================
// Pixel Format Types (Storage types)
// =============================================================================

/// Rgba8 pixel: bytes are [R, G, B, A] in memory order.
/// As a u32 on little-endian: 0xAABBGGRR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct Rgba8(pub u32);

/// Bgra8 pixel: bytes are [B, G, R, A] in memory order.
/// As a u32 on little-endian: 0xAARRGGBB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct Bgra8(pub u32);

impl Rgba8 {
    /// Creates a new RGBA pixel from component values.
    #[inline]
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self(u32::from_le_bytes([r, g, b, a]))
    }

    #[inline]
    pub fn r(self) -> u8 {
        self.0.to_le_bytes()[0]
    }
    #[inline]
    pub fn g(self) -> u8 {
        self.0.to_le_bytes()[1]
    }
    #[inline]
    pub fn b(self) -> u8 {
        self.0.to_le_bytes()[2]
    }
    #[inline]
    pub fn a(self) -> u8 {
        self.0.to_le_bytes()[3]
    }
}

impl Bgra8 {
    /// Creates a new BGRA pixel from component values.
    #[inline]
    pub fn new(b: u8, g: u8, r: u8, a: u8) -> Self {
        Self(u32::from_le_bytes([b, g, r, a]))
    }

    #[inline]
    pub fn b(self) -> u8 {
        self.0.to_le_bytes()[0]
    }
    #[inline]
    pub fn g(self) -> u8 {
        self.0.to_le_bytes()[1]
    }
    #[inline]
    pub fn r(self) -> u8 {
        self.0.to_le_bytes()[2]
    }
    #[inline]
    pub fn a(self) -> u8 {
        self.0.to_le_bytes()[3]
    }
}

// Swizzle: swap bytes 0 and 2 (R and B)
#[inline]
fn swizzle_rb(v: u32) -> u32 {
    (v & 0xFF00FF00) | ((v >> 16) & 0x000000FF) | ((v & 0x000000FF) << 16)
}

impl From<Bgra8> for Rgba8 {
    #[inline]
    fn from(bgra: Bgra8) -> Rgba8 {
        Rgba8(swizzle_rb(bgra.0))
    }
}

impl From<Rgba8> for Bgra8 {
    #[inline]
    fn from(rgba: Rgba8) -> Bgra8 {
        Bgra8(swizzle_rb(rgba.0))
    }
}

// =============================================================================
// Pixel Trait Implementations
// =============================================================================

impl Pixel for Rgba8 {
    #[inline]
    fn from_u32(v: u32) -> Self {
        Self(v)
    }
    #[inline]
    fn to_u32(self) -> u32 {
        self.0
    }
    #[inline]
    fn from_rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        let r = (r * 255.0).clamp(0.0, 255.0) as u8;
        let g = (g * 255.0).clamp(0.0, 255.0) as u8;
        let b = (b * 255.0).clamp(0.0, 255.0) as u8;
        let a = (a * 255.0).clamp(0.0, 255.0) as u8;
        Self::new(r, g, b, a)
    }
}

impl Pixel for Bgra8 {
    #[inline]
    fn from_u32(v: u32) -> Self {
        Self(v)
    }
    #[inline]
    fn to_u32(self) -> u32 {
        self.0
    }
    #[inline]
    fn from_rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        let r = (r * 255.0).clamp(0.0, 255.0) as u8;
        let g = (g * 255.0).clamp(0.0, 255.0) as u8;
        let b = (b * 255.0).clamp(0.0, 255.0) as u8;
        let a = (a * 255.0).clamp(0.0, 255.0) as u8;
        Self::new(b, g, r, a)
    }
}

// =============================================================================
// Platform-specific type aliases
// =============================================================================

/// Pixel format for X11 (XImage with ZPixmap on little-endian).
pub type X11Pixel = Bgra8;

/// Pixel format for Cocoa (CGImage with kCGImageAlphaPremultipliedLast).
pub type CocoaPixel = Rgba8;

/// Pixel format for Web (ImageData).
pub type WebPixel = Rgba8;

// =============================================================================
// Color Manifolds
// =============================================================================

/// The RGBA color cube as a terminal object.
///
/// Every color manifold factors through ColorCube via At:
///
/// ```text
///                    At<coords>
/// YourColorType  ─────────────────→  ColorCube  ───→  Discrete
/// ```
///
/// # Universal Property
///
/// For any manifold M that produces a color, there exists a unique
/// factorization M = At<ColorCube, r, g, b, a> where r,g,b,a are
/// the channel expressions.
///
/// # Generic Parameters
///
/// - `RedVar`: Coordinate variable for red channel (X for RGBA, Z for BGRA)
/// - `GreenVar`: Coordinate variable for green channel (Y for both)
/// - `BlueVar`: Coordinate variable for blue channel (Z for RGBA, X for BGRA)
/// - `AlphaVar`: Coordinate variable for alpha channel (W for both)
#[derive(Clone, Copy, Debug)]
pub struct ColorCube<RedVar, GreenVar, BlueVar, AlphaVar> {
    _phantom: core::marker::PhantomData<(RedVar, GreenVar, BlueVar, AlphaVar)>,
}

impl<R, G, B, A> Default for ColorCube<R, G, B, A> {
    fn default() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<R, G, B, A> Manifold<Field4> for ColorCube<R, G, B, A>
where
    R: Manifold<Field4, Output = Field> + Default,
    G: Manifold<Field4, Output = Field> + Default,
    B: Manifold<Field4, Output = Field> + Default,
    A: Manifold<Field4, Output = Field> + Default,
{
    type Output = Discrete;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Discrete {
        let (x, y, z, w) = p;
        // Evaluate each channel variable to extract the appropriate coordinate
        let r = R::default().eval_raw(x, y, z, w);
        let g = G::default().eval_raw(x, y, z, w);
        let b = B::default().eval_raw(x, y, z, w);
        let a = A::default().eval_raw(x, y, z, w);

        // pack() always produces RGBA byte order
        // Channel swapping happens via the variable mapping above
        Discrete::pack(r, g, b, a)
    }
}

/// RGBA byte order color cube (macOS, most platforms).
///
/// Maps coordinates to channels: X→Red, Y→Green, Z→Blue, W→Alpha
pub type RgbaColorCube = ColorCube<
    pixelflow_core::variables::X,
    pixelflow_core::variables::Y,
    pixelflow_core::variables::Z,
    pixelflow_core::variables::W,
>;

/// BGRA byte order color cube (Linux/X11).
///
/// Maps coordinates to channels: Z→Red, Y→Green, X→Blue, W→Alpha
pub type BgraColorCube = ColorCube<
    pixelflow_core::variables::Z,
    pixelflow_core::variables::Y,
    pixelflow_core::variables::X,
    pixelflow_core::variables::W,
>;

/// Platform-appropriate ColorCube (handles byte order based on target OS).
///
/// - macOS: RGBA byte order
/// - Linux: BGRA byte order (X11)
/// - Other: RGBA byte order (default)
#[cfg(target_os = "macos")]
pub type PlatformColorCube = RgbaColorCube;

#[cfg(target_os = "linux")]
pub type PlatformColorCube = BgraColorCube;

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub type PlatformColorCube = RgbaColorCube;

impl<RedVar, GreenVar, BlueVar, AlphaVar> ColorCube<RedVar, GreenVar, BlueVar, AlphaVar> {
    /// Create a color manifold from channel manifolds.
    ///
    /// This is a convenience constructor for `At<ColorCube, ...>`.
    ///
    /// # Example
    /// ```ignore
    /// use pixelflow_graphics::RgbaColorCube;
    /// use pixelflow_core::X;
    ///
    /// let grad = RgbaColorCube::default().at(X, 0.0, 0.0, 1.0);
    /// ```
    #[inline(always)]
    pub fn at<R, G, B, A>(self, r: R, g: G, b: B, a: A) -> At<R, G, B, A, Self> {
        At {
            inner: self,
            x: r,
            y: g,
            z: b,
            w: a,
        }
    }
}

/// Grayscale: lifts a scalar to R=G=B, A=1.
///
/// Convenience for the common pattern:
/// ```ignore
/// At { inner: RgbaColorCube, x: v, y: v, z: v, w: 1.0 }
/// ```
///
/// Defaults to RGBA byte order. For platform-specific byte order, use the
/// runtime's PlatformColorCube re-export.
pub type Grayscale<M> = At<M, M, M, Field, RgbaColorCube>;

/// Create a grayscale manifold (RGBA byte order).
#[allow(non_snake_case)]
#[inline(always)]
pub fn Grayscale<M: Manifold + Clone>(m: M) -> Grayscale<M> {
    At {
        inner: RgbaColorCube::default(),
        x: m.clone(),
        y: m.clone(),
        z: m.clone(),
        w: Field::from(1.0),
    }
}

/// Composes 4 Field manifolds into a single RGBA output.
///
/// Type alias for `At<R, G, B, A, RgbaColorCube>`.
///
/// Defaults to RGBA byte order. For platform-specific byte order, use the
/// runtime's PlatformColorCube re-export.
pub type ColorManifold<R, G, B, A> = At<R, G, B, A, RgbaColorCube>;

/// Create a new color manifold from four channel manifolds (RGBA byte order).
///
/// This replaces `color_manifold`.
#[inline(always)]
pub fn color_manifold<R, G, B, A>(r: R, g: G, b: B, a: A) -> ColorManifold<R, G, B, A> {
    At {
        inner: RgbaColorCube::default(),
        x: r,
        y: g,
        z: b,
        w: a,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgba8_components() {
        let c = Rgba8::new(0x11, 0x22, 0x33, 0xFF);
        assert_eq!(c.r(), 0x11);
        assert_eq!(c.g(), 0x22);
        assert_eq!(c.b(), 0x33);
        assert_eq!(c.a(), 0xFF);
    }

    #[test]
    fn test_bgra8_components() {
        let c = Bgra8::new(0x33, 0x22, 0x11, 0xFF);
        assert_eq!(c.b(), 0x33);
        assert_eq!(c.g(), 0x22);
        assert_eq!(c.r(), 0x11);
        assert_eq!(c.a(), 0xFF);
    }

    #[test]
    fn test_rgba8_to_bgra8() {
        let rgba = Rgba8::new(0x11, 0x22, 0x33, 0xFF);
        let bgra = Bgra8::from(rgba);
        assert_eq!(bgra.r(), 0x11);
        assert_eq!(bgra.g(), 0x22);
        assert_eq!(bgra.b(), 0x33);
        assert_eq!(bgra.a(), 0xFF);
    }

    #[test]
    fn test_named_color_manifold() {
        use pixelflow_core::{materialize_discrete, PARALLELISM};
        let red = NamedColor::Red;
        let mut out = vec![0u32; PARALLELISM];
        materialize_discrete(&red, 0.0, 0.0, &mut out);

        let val = out[0];
        let r = (val & 0xFF) as f32 / 255.0;
        let g = ((val >> 8) & 0xFF) as f32 / 255.0;
        let b = ((val >> 16) & 0xFF) as f32 / 255.0;
        let a = ((val >> 24) & 0xFF) as f32 / 255.0;

        // Red is (205, 0, 0) -> (0.8039, 0, 0)
        assert!((r - 205.0 / 255.0).abs() < 1e-2);
        assert!((g - 0.0).abs() < 1e-2);
        assert!((b - 0.0).abs() < 1e-2);
        assert!((a - 1.0).abs() < 1e-2);
    }

    #[test]
    fn test_color_manifold() {
        use pixelflow_core::{materialize_discrete, PARALLELISM};
        let c = Color::Rgb(10, 20, 30);
        let mut out = vec![0u32; PARALLELISM];
        materialize_discrete(&c, 0.0, 0.0, &mut out);

        let val = out[0];
        let r = (val & 0xFF) as f32 / 255.0;
        let g = ((val >> 8) & 0xFF) as f32 / 255.0;
        let b = ((val >> 16) & 0xFF) as f32 / 255.0;
        let a = ((val >> 24) & 0xFF) as f32 / 255.0;

        assert!((r - 10.0 / 255.0).abs() < 1e-2);
        assert!((g - 20.0 / 255.0).abs() < 1e-2);
        assert!((b - 30.0 / 255.0).abs() < 1e-2);
        assert!((a - 1.0).abs() < 1e-2);
    }
}
