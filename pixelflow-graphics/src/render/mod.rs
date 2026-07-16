pub mod aa;
pub mod color;
pub mod frame;
pub mod pixel;
pub mod rasterizer;

#[cfg(test)]
mod pict; // PICT-style pairwise covering-array generator (POC)
#[cfg(test)]
mod pict_color_tests; // Pairwise color/pixel testing built on `pict`

pub use aa::aa_coverage;
pub use color::{
    AttrFlags, Bgra8, BgraColorCube, CocoaPixel, Color, ColorCube, ColorManifold, Grayscale,
    NamedColor, Rgba8, RgbaColorCube, WebPixel, X11Pixel,
};
pub use frame::Frame;
pub use pixel::Pixel;
pub use rasterizer::rasterize;
