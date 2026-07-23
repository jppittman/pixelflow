// Probe: AnalyticalLine lowers to IR that (after the runtime calculus) matches
// the combinator-over-Jet2 coverage — the leaf-delegation path for P5.
use pixelflow_core::jet::Jet2;
use pixelflow_core::{Field, Lower, LowerEnv, Manifold};
use pixelflow_graphics::fonts::ttf_curve_analytical::AnalyticalLine;
use pixelflow_ir::arena::ExprArena;
use pixelflow_ir::backend::emit::lowering::lower_dwrt_owned;
use pixelflow_ir::binding::BindingTable;
use pixelflow_ir::eval::eval_scalar;

fn lane0(f: Field) -> f32 {
    unsafe { core::mem::transmute_copy(&f) }
}

/// A leaf's lowered IR (after the runtime calculus) must match its
/// combinator-over-Jet2 coverage — the leaf-delegation contract for P5.
fn assert_leaf_matches<M>(seg: &M, name: &str, pts: &[(f32, f32)])
where
    M: Lower
        + Manifold<(Jet2, Jet2, Jet2, Jet2), Output = Field>,
{
    let mut arena = ExprArena::new();
    let mut env = LowerEnv::default();
    let root = seg.lower(&mut arena, &mut env).expect("leaf lowers");
    let (lowered, lroot) = lower_dwrt_owned(&arena, root).expect("calculus");

    for &(x, y) in pts {
        let ir = eval_scalar(&lowered, lroot, &[x, y, 0.0, 0.0], &BindingTable::empty());
        let comb = lane0(Manifold::eval(
            seg,
            (
                Jet2::x(Field::from(x)),
                Jet2::y(Field::from(y)),
                Jet2::constant(Field::from(0.0)),
                Jet2::constant(Field::from(0.0)),
            ),
        ));
        assert!(
            (ir - comb).abs() < 1e-3,
            "{name} at ({x},{y}): lowered {ir} vs combinator/Jet2 {comb}"
        );
    }
}

#[test]
fn line_lower_matches_combinator_jet2() {
    let line = AnalyticalLine::from_points([2.0, 1.0], [8.0, 9.0]).unwrap();
    assert_leaf_matches(&line, "line", &[(4.0, 3.0), (5.5, 6.0), (3.0, 8.0), (7.0, 2.0)]);
}

#[test]
fn quad_lower_matches_combinator_jet2() {
    use pixelflow_graphics::fonts::ttf::make_quad;
    let quad = make_quad([[1.0, 1.0], [5.0, 8.0], [9.0, 2.0]]);
    // make_quad returns Quad<AnalyticalQuad>; lower/eval delegate to the kernel.
    assert_leaf_matches(&quad, "quad", &[(3.0, 3.0), (5.0, 5.0), (7.0, 4.0), (2.0, 6.0)]);
}
