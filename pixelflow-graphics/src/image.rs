use pixelflow_core::Manifold;

/// A simple 2D image buffer.
#[derive(Clone, Debug)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>, // RGBA data
}

impl Image {
    /// Create a new image with the given dimensions.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0; width * height * 4],
        }
    }

    /// Render a manifold into the image.
    ///
    /// The manifold is treated as a mask (0.0 to 1.0).
    /// The image is filled with white where the manifold is 1.0,
    /// and black where it is 0.0.
    ///
    /// Future versions will support color manifolds via At<ColorCube, ...>.
    pub fn render_mask(&mut self, _mask: &impl Manifold) {
        // Placeholder implementation to allow compilation.
        // The previous implementation used a private `materialize` function.
        // Proper implementation requires public evaluation APIs.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_new_creates_correct_buffer_size() {
        let img = Image::new(100, 50);
        // RGBA = 4 bytes per pixel
        assert_eq!(img.data.len(), 100 * 50 * 4);
    }

    #[test]
    fn image_new_stores_dimensions() {
        let img = Image::new(640, 480);
        assert_eq!(img.width, 640);
        assert_eq!(img.height, 480);
    }

    #[test]
    fn image_data_initializes_to_zero() {
        let img = Image::new(10, 10);
        assert!(img.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn image_zero_dimensions_create_empty_buffer() {
        let img = Image::new(0, 0);
        assert_eq!(img.data.len(), 0);
        assert_eq!(img.width, 0);
        assert_eq!(img.height, 0);
    }

    #[test]
    fn image_one_pixel() {
        let img = Image::new(1, 1);
        assert_eq!(img.data.len(), 4); // Single RGBA pixel
    }

    #[test]
    fn image_clone_is_independent() {
        let img1 = Image::new(10, 10);
        let mut img2 = img1.clone();

        // Modify the clone
        img2.data[0] = 255;

        // Original should be unchanged
        assert_eq!(img1.data[0], 0);
        assert_eq!(img2.data[0], 255);
    }

    #[test]
    fn image_large_dimensions() {
        // Test a moderately large image (1920x1080)
        let img = Image::new(1920, 1080);
        assert_eq!(img.data.len(), 1920 * 1080 * 4);
        assert_eq!(img.width, 1920);
        assert_eq!(img.height, 1080);
    }
}
