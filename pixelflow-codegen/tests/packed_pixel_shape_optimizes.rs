//! The packed-pixel kernel shape must optimize, not bail: four channel
//! lanes (gather → scale → `TruncToInt` → `Shl` → or-fold) share their
//! coordinate geometry only if the integer-domain ops are representable in
//! the runtime e-graph. This bailed silently before the int ops joined
//! `runtime_op_from_kind` — and the frame benchmark read 1.01x against the
//! four-plane path because BOTH compiled unoptimized.
use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{BufferDecl, BufferIdentity, ExprArena, ExprId};

fn reachable(arena: &ExprArena, root: ExprId) -> usize {
    let mut seen = vec![false; arena.len()];
    let mut stack = vec![root];
    let mut n = 0;
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        n += 1;
        stack.extend(arena.children(id));
    }
    n
}

#[test]
fn packed_shape_gets_cse() {
    // Four channel fragments sharing geometry, spliced into one root —
    // the packed kernel's shape at reduced scale.
    let mut a = ExprArena::new();
    let cells = a.declare_buffer(BufferDecl {
        id: BufferIdentity::mint(),
        width: 100,
        height: 10,
    });
    let x = a.push_var(0);
    let y = a.push_var(1);
    let ten = a.push_const(10.0);
    let mut lanes = Vec::new();
    for c in 0..4u32 {
        // each lane duplicates the shared geometry (col = floor(x/10)*10 …)
        let inv = a.push_const(0.1);
        let xc = a.push_binary(OpKind::Mul, x, inv);
        let col = a.push_unary(OpKind::Floor, xc);
        let cx = a.push_binary(OpKind::Mul, col, ten);
        let off = a.push_const(2.0 + c as f32);
        let idx = a.push_binary(OpKind::Add, cx, off);
        let g = a.push_gather(cells, idx, y);
        let f255 = a.push_const(255.0);
        let scaled = a.push_binary(OpKind::Mul, g, f255);
        let t = a.push_unary(OpKind::TruncToInt, scaled);
        let sh = a.push_const((c * 8) as f32);
        let shifted = a.push_binary(OpKind::Shl, t, sh);
        lanes.push(shifted);
    }
    let or1 = a.push_binary(OpKind::BitOr, lanes[0], lanes[1]);
    let or2 = a.push_binary(OpKind::BitOr, or1, lanes[2]);
    let root = a.push_binary(OpKind::BitOr, or2, lanes[3]);

    let before = reachable(&a, root);
    let out = pixelflow_search::runtime::optimize_runtime_arena(
        &a,
        root,
        pixelflow_ir::LatticeShape::POINT,
    );
    match out {
        None => panic!("optimizer BAILED on the packed shape ({before} nodes)"),
        Some(res) => {
            let (oa, or_) = (&res.0, res.1);
            let after = reachable(oa, or_);
            println!("before={before} after={after}");
            assert!(after < before, "no CSE: {before} -> {after}");
        }
    }
}
