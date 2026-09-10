//! The packed-pixel kernel shape must optimize, not bail: four channel
//! lanes (gather → scale → `TruncToInt` → `Shl` → or-fold) share their
//! coordinate geometry only if the integer-domain ops are representable in
//! the runtime e-graph. This bailed silently before the int ops joined
//! `runtime_op_from_kind` — and the frame benchmark read 1.01x against the
//! four-plane path because BOTH compiled unoptimized.
use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{BufferDecl, BufferIdentity};
use pixelflow_ir::{ExprBuilder, Rooted};

fn reachable(root: pixelflow_ir::Node<'_, pixelflow_ir::ExprData>) -> usize {
    root.node_count()
}

#[test]
fn packed_shape_gets_cse() {
    // Four channel fragments sharing geometry, spliced into one root —
    // the packed kernel's shape at reduced scale.
    let mut a = ExprBuilder::new();
    let cells = a.buffer(BufferDecl {
        id: BufferIdentity::mint(),
        width: 100,
        height: 10,
    });
    let x = a.var(0);
    let y = a.var(1);
    let ten = a.constant(10.0);
    let mut lanes = Vec::new();
    for c in 0..4u32 {
        // each lane duplicates the shared geometry (col = floor(x/10)*10 …)
        let inv = a.constant(0.1);
        let xc = a.binary(OpKind::Mul, x, inv);
        let col = a.unary(OpKind::Floor, xc);
        let cx = a.binary(OpKind::Mul, col, ten);
        let off = a.constant(2.0 + c as f32);
        let idx = a.binary(OpKind::Add, cx, off);
        let g = a.ternary(OpKind::Gather, cells, idx, y);
        let f255 = a.constant(255.0);
        let scaled = a.binary(OpKind::Mul, g, f255);
        let t = a.unary(OpKind::TruncToInt, scaled);
        let sh = a.constant((c * 8) as f32);
        let shifted = a.binary(OpKind::Shl, t, sh);
        lanes.push(shifted);
    }
    let or1 = a.binary(OpKind::BitOr, lanes[0], lanes[1]);
    let or2 = a.binary(OpKind::BitOr, or1, lanes[2]);
    let root = a.binary(OpKind::BitOr, or2, lanes[3]);
    let graph = a.finish_one(root);

    let before = reachable(graph.root());
    let (legacy, legacy_root) = graph.root().marshal(graph.environment());
    let out = pixelflow_search::runtime::optimize_runtime_arena(
        &legacy,
        legacy_root,
        pixelflow_ir::LatticeShape::POINT,
    );
    match out {
        None => panic!("optimizer BAILED on the packed shape ({before} nodes)"),
        Some(res) => {
            let after = reachable(Rooted::unmarshal(&res.0, &[res.1]).0.entry());
            println!("before={before} after={after}");
            assert!(after < before, "no CSE: {before} -> {after}");
        }
    }
}
