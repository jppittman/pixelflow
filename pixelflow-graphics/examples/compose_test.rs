//! Testing kernel composition patterns
use pixelflow_core::{Field, Manifold};
use pixelflow_compiler::kernel;

type Field4 = (Field, Field, Field, Field);

fn field4(x: f32, y: f32) -> Field4 {
    (Field::from(x), Field::from(y), Field::from(0.0), Field::from(0.0))
}

fn main() {
    // Pattern 1: Parameterized kernel returns a manifold
    let dist = kernel!(|cx: f32, cy: f32| {
        let dx = X - cx;
        let dy = Y - cy;
        (dx * dx + dy * dy).sqrt()
    });

    // Instantiate with concrete parameters
    let d = dist(1.0, 2.0);

    // Evaluate at a point
    let p = field4(1.5, 2.0);
    let result = d.eval(p);
    println!("distance from (1.0, 2.0) at (1.5, 2.0): {:?}", result);

    // Pattern 2: Circle with radius built into the kernel
    let circle = kernel!(|cx: f32, cy: f32, r: f32| {
        let dx = X - cx;
        let dy = Y - cy;
        (dx * dx + dy * dy).sqrt() - r
    });

    let c = circle(1.0, 2.0, 0.5);
    let result2 = c.eval(p);
    println!("circle(center=(1.0, 2.0), r=0.5) at (1.5, 2.0): {:?}", result2);
}
