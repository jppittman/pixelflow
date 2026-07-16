//! Test: Parallel Rasterizer coverage
//!
//! Validates that `render_parallel` and `render_work_stealing` produce identical output
//! to single-threaded execution, and handle edge cases (small height, odd dimensions).

use pixelflow_core::{Field, Manifold};
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::parallel::{
    render_parallel, render_work_stealing, RenderOptions,
};
use pixelflow_graphics::render::rasterizer::rasterize;
use pixelflow_graphics::render::color::Rgba8;

// A simple test manifold: Gradient X + Y
#[derive(Copy, Clone)]
struct TestGradient;

impl Manifold<(Field, Field, Field, Field)> for TestGradient {
    type Output = pixelflow_core::Discrete;

    fn eval(&self, p: (Field, Field, Field, Field)) -> Self::Output {
        let (x, y, _, _) = p;
        // Evaluate AST to get Field values
        // Note: Field ops return AST nodes, so we must call .eval() to get the result.
        // Since operands are concrete Fields, we can pass dummy coordinates.
        let dummy: (Field, Field, Field, Field) = (Field::default(), Field::default(), Field::default(), Field::default());

        // Simple gradient: (x + y) * 0.1
        let val = ((x + y) * Field::from(0.1)).eval(dummy);

        let r = val;
        let g = (val * Field::from(0.5)).eval(dummy);
        let b = (val * Field::from(0.2)).eval(dummy);

        pixelflow_core::Discrete::pack(r, g, b, Field::from(1.0))
    }
}

// Helper to create a frame and render single-threaded reference
fn render_reference(width: u32, height: u32) -> Frame<Rgba8> {
    let mut frame: Frame<Rgba8> = Frame::new(width, height);
    rasterize(&TestGradient, &mut frame, 1);
    frame
}

#[test]
fn render_parallel_matches_single_threaded_output() {
    let width = 100;
    let height = 100;
    let reference = render_reference(width, height);

    let mut frame: Frame<Rgba8> = Frame::new(width, height);
    let options = RenderOptions { num_threads: 4 };

    // Target: render_parallel
    render_parallel(&TestGradient, &mut frame, options);

    assert_eq!(frame.data, reference.data, "render_parallel output mismatch");
}

#[test]
fn render_parallel_matches_single_threaded_output_odd_threads() {
    let width = 100;
    let height = 100;
    let reference = render_reference(width, height);

    let mut frame: Frame<Rgba8> = Frame::new(width, height);
    let options = RenderOptions { num_threads: 3 };

    render_parallel(&TestGradient, &mut frame, options);

    assert_eq!(frame.data, reference.data, "render_parallel (3 threads) output mismatch");
}

#[test]
fn render_parallel_handles_small_height() {
    // Height < num_threads
    let width = 50;
    let height = 2;
    let reference = render_reference(width, height);

    let mut frame: Frame<Rgba8> = Frame::new(width, height);
    let options = RenderOptions { num_threads: 4 };

    render_parallel(&TestGradient, &mut frame, options);

    assert_eq!(frame.data, reference.data, "render_parallel small height mismatch");
}

#[test]
fn render_parallel_handles_height_one() {
    // Height = 1
    let width = 50;
    let height = 1;
    let reference = render_reference(width, height);

    let mut frame: Frame<Rgba8> = Frame::new(width, height);
    let options = RenderOptions { num_threads: 4 };

    // Note: implementation might fallback to single threaded if height=1
    // but we test the interface contract.
    render_parallel(&TestGradient, &mut frame, options);

    assert_eq!(frame.data, reference.data, "render_parallel height=1 mismatch");
}

#[test]
fn render_work_stealing_matches_single_threaded_output() {
    let width = 100;
    let height = 100;
    let reference = render_reference(width, height);

    let mut frame: Frame<Rgba8> = Frame::new(width, height);
    let options = RenderOptions { num_threads: 4 };

    render_work_stealing(&TestGradient, &mut frame, options);

    assert_eq!(frame.data, reference.data, "render_work_stealing output mismatch");
}

#[test]
fn render_work_stealing_handles_height_one() {
    let width = 50;
    let height = 1;
    let reference = render_reference(width, height);

    let mut frame: Frame<Rgba8> = Frame::new(width, height);
    let options = RenderOptions { num_threads: 4 };

    render_work_stealing(&TestGradient, &mut frame, options);

    assert_eq!(frame.data, reference.data, "render_work_stealing height=1 mismatch");
}
