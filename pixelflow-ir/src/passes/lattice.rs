//! The lattice's own two folds, as legalize passes.
//!
//! `collapse(extent)` wraps a kernel in the row/column/lane folds a
//! collapse *is*; `pack(lanes)` strip-mines the column fold to a target's
//! lane width. Both are arena-level transforms, and
//! [`legalize`](super::legalize) runs them between `lower_dwrt` and
//! `expand_gather` — see [`super`]'s module doc for why they sit there in the
//! full pass order.
//!
//! ```text
//! collapse(f) = fold_{j∈[0,h)} fold_{i∈[0,w)} fold_{l∈[0,1)}
//!                  Write(row=j, col=i, lane=l, f(x0 + i + l, y0 + j))
//! pack(L)     = fold_j (   fold_{i∈[0,w−r) step L}   fold_{l∈[0,L)} Write(…)
//!                        ; fold_{i∈[w−r,w−r+1)}      fold_{l∈[0,r)} Write(…) )
//!                                                      (the `;` arm only when r > 0)
//! ```
//!
//! `w × h` is the extent, `q = w / L`, `r = w % L`; every fold is over
//! [`Monoid::SEQ`], the unit monoid; `;` is [`OpKind::Seq`]. `collapse` builds
//! a *degenerate* `[0,1)` lane fold rather than none at all, so `Write`
//! always has all three binders and `pack` only ever reshapes the two inner
//! folds' ranges — the body, the `Write` node itself, is shared and
//! untouched. This is the monadic reading — a `SEQ` fold is `for_`, `Seq` is
//! `>>` — and `pack` is exactly the chunking law
//! `for_ [0,w) f = for_ [0,w−r) step L (\i -> for_ [0,L) (\l -> f(i+l))) >>
//! for_ [w−r] (\i -> for_ [0,r) (\l -> f(i+l)))`
//! (docs/plans/2026-09-16-collapse-is-a-fold.md §2.1–§2.4).

use crate::arena::{COORD_AXES, ExprArena, ExprId, ExprNode, UniformDecl};
use crate::fold::{Binder, Fold, Monoid, RangeFold};
use crate::kind::OpKind;
use crate::variance::LatticeShape;

/// What [`legalize`](super::legalize) wraps a kernel for: the domain it
/// tabulates, and the lane width the target executes the innermost fold by.
///
/// The lanes are the one thing codegen tells the IR about a target, and they
/// are a parameter rather than a constant because `pixelflow-ir` never names
/// a width (CLAUDE.md, "SIMD is an implementation detail").
#[derive(Clone, Copy, Debug)]
pub struct Collapse {
    /// The extent and the origin.
    pub domain: Domain,
    /// Lanes per batch: [`pack`]'s `L`.
    pub lanes: u32,
}

/// The domain a collapse tabulates a kernel over: its extent, and the origin
/// the call supplies. `LatticeShape` is "the compile-time half of
/// `pixelflow_core::Lattice`, origin erased"; this puts the origin back as
/// two uniforms.
#[derive(Clone, Copy, Debug)]
pub struct Domain {
    /// Samples per axis, `[w, h]`: the two outer folds' trip counts.
    pub shape: LatticeShape,
    /// `[x0, y0]`: the lattice position of sample `(0, 0)`, bound per call.
    pub origin: [UniformDecl; COORD_AXES],
}

/// Wrap the kernel at `root` in the lattice's three folds — row, column,
/// lane — around one [`Write`](ExprNode::Write), and return the new root
/// (the row fold) in the same arena. Target-agnostic: no lane width appears
/// (that is [`pack`]'s one parameter).
///
/// `f ↦ fold_j fold_i fold_l Write(row=j, col=i, lane=l, f.at(x0+i+l, y0+j))`,
/// the lane fold degenerate at `[0,1)` so `Write` always has all three
/// binders and [`pack`] only ever reshapes ranges (module doc). `x0`/`y0` are
/// `domain.origin`, declared as uniforms; the warped `x` is built
/// `Add(Add(Uniform(x0), Var(col)), Var(lane))` — `(x0 + col)` is
/// lane-uniform and hoists to the batch scope — and `y` is
/// `Add(Uniform(y0), Var(row))`. Building the warp this way and substituting
/// it for `Var(0)`/`Var(1)` is exactly [`Kernel::at`](crate::Kernel::at).
///
/// # Panics
///
/// - If a `Ref`, a `Guard`, a `Dwrt` (any arity), or an interval fold is
///   reachable from `root`. A `Ref`/`Guard` is a name:
///   [`substitute_vars_with`](ExprArena::substitute_vars_with) copies it
///   through without reaching the referent's `Var(0)`, so the warp below
///   would silently never reach it. A `Dwrt` must be taken with respect to
///   `Var(0)` before this pass substitutes that variable away, or the
///   chain-rule factor it needs is gone. An interval fold is an integral,
///   and an integral is not a loop: nothing downstream can run one, so it
///   must already have been replaced by its quadrature. Run `expand_refs`,
///   then [`resolve`](super::resolve) (intervals, then `Dwrt`), first — see
///   [`legalize`](super::legalize) for the order.
/// - If fewer than three binder slots are free — neither bound by a
///   reachable `Reduce` nor read by a reachable `Var` in the binder range: a
///   kernel already nesting three folds of its own leaves nothing for the
///   lattice.
pub fn collapse(arena: &mut ExprArena, root: ExprId, domain: Domain) -> ExprId {
    let taken = reachable_taken_binders(arena, root);
    let mut free = (0..Binder::COUNT as u8).filter(|&slot| !taken[slot as usize]);
    let mut next_binder = |which: &str| {
        let slot = free.next().unwrap_or_else(|| {
            panic!(
                "collapse: no free binder slot for the {which} fold — a \
                 kernel already nesting three folds of its own leaves \
                 nothing for the lattice"
            )
        });
        Binder::from_slot(slot).expect("free() only ever yields a slot below Binder::COUNT")
    };
    let row = next_binder("row");
    let col = next_binder("col");
    let lane = next_binder("lane");

    let x0_id = arena.uniform_slot_for(domain.origin[0]);
    let y0_id = arena.uniform_slot_for(domain.origin[1]);
    let x0 = arena.push_uniform(x0_id);
    let y0 = arena.push_uniform(y0_id);

    let i = arena.push_var(col.var());
    let l = arena.push_var(lane.var());
    let j = arena.push_var(row.var());

    // `(x0 + i)` first, then `+ l`: the inner sum reads no lane index, so it
    // is lane-uniform and hoists to the batch scope by the same variance bit
    // step 6 reads to choose a broadcast load over a gather.
    let x0_plus_i = arena.push_binary(OpKind::Add, x0, i);
    let x = arena.push_binary(OpKind::Add, x0_plus_i, l);
    let y = arena.push_binary(OpKind::Add, y0, j);

    let body = arena.substitute_vars_with(root, &[(0, x), (1, y)]);
    let write = arena.push_write(row, col, lane, body);

    let [w, h] = domain.shape.extent();
    let lane_fold = arena.push_reduce(Fold::new(Monoid::SEQ, lane, 0..1), write);
    let col_fold = arena.push_reduce(Fold::new(Monoid::SEQ, col, 0..w), lane_fold);
    arena.push_reduce(Fold::new(Monoid::SEQ, row, 0..h), col_fold)
}

/// The full pass order, named once so every panic message in this file spells
/// it identically.
const PASS_ORDER: &str = "expand_refs -> expand_intervals -> lower_dwrt -> collapse -> pack \
     -> expand_gather -> expand_transcendentals";

/// Which binder slots [`collapse`] must not choose — bound by a reachable
/// `Reduce`, or read by a reachable `Var` in the binder range — found in one
/// walk of `root`'s reachable subgraph that also refuses, by panicking and
/// naming [`PASS_ORDER`], any reachable `Ref`, `Guard`, `Dwrt`, or interval
/// fold (see [`collapse`]'s own doc for why each is refused).
fn reachable_taken_binders(arena: &ExprArena, root: ExprId) -> [bool; Binder::COUNT] {
    let mut taken = [false; Binder::COUNT];
    let mut seen = alloc::vec![false; arena.len()];
    let mut stack = alloc::vec![root];
    while let Some(id) = stack.pop() {
        let idx = id.0 as usize;
        if core::mem::replace(&mut seen[idx], true) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Ref(key) => panic!(
                "collapse: {key:?} is a Ref reachable from root — a name, \
                 and substitute_vars_with copies it through without \
                 reaching the referent's Var(0), so the coordinate warp \
                 would silently never reach it; run expand_refs first \
                 ({PASS_ORDER})"
            ),
            ExprNode::Guard { .. } => panic!(
                "collapse: a Guard is reachable from root — a name, for the \
                 same reason as a Ref above; run expand_refs first \
                 ({PASS_ORDER})"
            ),
            ExprNode::Unary(OpKind::Dwrt, _)
            | ExprNode::Binary(OpKind::Dwrt, _, _)
            | ExprNode::Ternary(OpKind::Dwrt, _, _, _)
            | ExprNode::Nary(OpKind::Dwrt, _) => panic!(
                "collapse: a Dwrt is reachable from root — it must be taken \
                 with respect to Var(0) before this pass substitutes that \
                 variable away; run lower_dwrt first ({PASS_ORDER})"
            ),
            ExprNode::Reduce {
                fold: Fold::Interval(interval),
                ..
            } => panic!(
                "collapse: an interval fold ({}) is reachable from root — an \
                 integral, which no loop nest can run; it must be closed by \
                 a rule or replaced by its quadrature before the lattice \
                 wraps the kernel; run expand_intervals first ({PASS_ORDER})",
                Fold::Interval(interval)
            ),
            ExprNode::Var(i) => {
                if let Some(b) = Binder::from_var(i) {
                    taken[b.slot() as usize] = true;
                }
            }
            ExprNode::Reduce { fold, .. } => {
                taken[fold.binder().slot() as usize] = true;
            }
            _ => {}
        }
        stack.extend(arena.children(id));
    }
    taken
}

/// Strip-mine [`collapse`]'s column fold by `lanes`: a main fold stepping by
/// `lanes` over `[0, w−r)` with a full lane fold `[0, lanes)` inside it, then
/// — only when `r = w mod lanes` is nonzero — the remainder
/// `[w−r, w−r+1)` × `[0, r)`, sequenced after the main fold under the row
/// fold with [`OpKind::Seq`]. The `Write` [`collapse`] built is the same
/// `ExprId` in every arm: only the two inner folds' ranges move, never the
/// body they wrap (module doc — the chunking law for `for_`).
///
/// When `w < lanes` this still holds: the main fold becomes the empty strided
/// fold `[0,0)`, and the whole column is the remainder. No special case.
///
/// # Panics
///
/// - If `root` is not exactly the shape [`collapse`] builds — three nested
///   `Reduce`s over [`Monoid::SEQ`] (row, then col, then a lane fold over
///   `[0,1)`) wrapping a `Write` whose `row`/`col`/`lane` equal the three
///   folds' binders, in that order. Naming a pipeline-order bug: `pack` must
///   run directly on `collapse`'s own output.
/// - If `lanes == 0`.
pub fn pack(arena: &mut ExprArena, root: ExprId, lanes: u32) -> ExprId {
    assert!(lanes != 0, "pack: lanes must be nonzero");

    let (row_fold, col_fold, lane_fold, write) = collapse_shape(arena, root);
    let row = row_fold.binder();
    let col = col_fold.binder();
    let lane = lane_fold.binder();

    let w = col_fold.range().end;
    let h = row_fold.range().end;
    let r = w % lanes;

    let lane_main = arena.push_reduce(Fold::new(Monoid::SEQ, lane, 0..lanes), write);
    let col_main = arena.push_reduce(
        Fold::strided(Monoid::SEQ, col, 0..(w - r), lanes),
        lane_main,
    );

    let body = if r > 0 {
        let lane_rem = arena.push_reduce(Fold::new(Monoid::SEQ, lane, 0..r), write);
        let col_rem =
            arena.push_reduce(Fold::new(Monoid::SEQ, col, (w - r)..(w - r + 1)), lane_rem);
        arena.push_binary(OpKind::Seq, col_main, col_rem)
    } else {
        col_main
    };

    arena.push_reduce(Fold::new(Monoid::SEQ, row, 0..h), body)
}

/// Read [`collapse`]'s three folds and its shared `Write` back out of `root`,
/// or panic naming the shape `pack` expects — see [`pack`]'s own doc.
fn collapse_shape(arena: &ExprArena, root: ExprId) -> (RangeFold, RangeFold, RangeFold, ExprId) {
    const SHAPE: &str = "pack expects collapse's shape: three nested Reduces \
        over Monoid::SEQ — row, then col, then a lane fold over [0,1) — \
        wrapping a Write whose row/col/lane equal the three folds' binders, \
        in that order";

    // A range, each of them: `collapse` built them, and it refuses an
    // interval anywhere below.
    let ExprNode::Reduce {
        fold: Fold::Range(row_fold),
        body: col_id,
    } = arena.node(root)
    else {
        panic!("{SHAPE} (root is not a range Reduce)");
    };

    let ExprNode::Reduce {
        fold: Fold::Range(col_fold),
        body: lane_id,
    } = arena.node(col_id)
    else {
        panic!("{SHAPE} (the row fold's body is not a range Reduce)");
    };

    let ExprNode::Reduce {
        fold: Fold::Range(lane_fold),
        body: write,
    } = arena.node(lane_id)
    else {
        panic!("{SHAPE} (the col fold's body is not a range Reduce)");
    };

    let ExprNode::Write {
        row: w_row,
        col: w_col,
        lane: w_lane,
        ..
    } = arena.node(write)
    else {
        panic!("{SHAPE} (the lane fold's body is not a Write)");
    };
    let write_binders = (w_row, w_col, w_lane);

    for fold in [row_fold, col_fold, lane_fold] {
        assert_eq!(
            fold.monoid(),
            Monoid::SEQ,
            "{SHAPE} (every fold must be over Monoid::SEQ)"
        );
    }
    assert_eq!(
        row_fold.range().start,
        0,
        "{SHAPE} (the row fold must start at 0)"
    );
    assert_eq!(
        col_fold.range().start,
        0,
        "{SHAPE} (the col fold must start at 0)"
    );
    assert_eq!(
        lane_fold.range(),
        0..1,
        "{SHAPE} (the lane fold must be collapse's degenerate [0,1))"
    );
    assert_eq!(
        write_binders,
        (row_fold.binder(), col_fold.binder(), lane_fold.binder()),
        "{SHAPE} (the Write's binders must equal the three folds', in order)"
    );

    (row_fold, col_fold, lane_fold, write)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{BufferDecl, BufferIdentity, UniformIdentity};
    use crate::variance::{Variance, compute_arena_variance};
    use crate::{Kernel, key::KernelKey};

    /// Two uniforms, freshly minted, for `Domain::origin` — no test cares
    /// about their default beyond "some value", so `0.0` throughout.
    fn mint_origin() -> [UniformDecl; COORD_AXES] {
        [
            UniformDecl {
                id: UniformIdentity::mint(),
                default: 0.0,
            },
            UniformDecl {
                id: UniformIdentity::mint(),
                default: 0.0,
            },
        ]
    }

    /// Whether `Var(target)` is reachable from `root` — the check
    /// `variance.rs`'s `six_nested_binders_each_take_a_slot` uses to confirm
    /// no rename captured an index it should not have.
    fn reachable_var(arena: &ExprArena, root: ExprId, target: u8) -> bool {
        let mut seen = alloc::vec![false; arena.len()];
        let mut stack = alloc::vec![root];
        while let Some(id) = stack.pop() {
            let idx = id.0 as usize;
            if core::mem::replace(&mut seen[idx], true) {
                continue;
            }
            if matches!(arena.node(id), ExprNode::Var(v) if v == target) {
                return true;
            }
            stack.extend(arena.children(id));
        }
        false
    }

    /// Whether an `OpKind::Seq` node is reachable from `root`.
    fn reachable_seq(arena: &ExprArena, root: ExprId) -> bool {
        let mut seen = alloc::vec![false; arena.len()];
        let mut stack = alloc::vec![root];
        while let Some(id) = stack.pop() {
            let idx = id.0 as usize;
            if core::mem::replace(&mut seen[idx], true) {
                continue;
            }
            if matches!(arena.node(id), ExprNode::Binary(OpKind::Seq, _, _)) {
                return true;
            }
            stack.extend(arena.children(id));
        }
        false
    }

    /// `X * 2 + Y` at a `5 × 3` lattice: the exact shape `collapse` must
    /// build, checked node by node, plus the two side effects — the two
    /// uniforms, and that no coordinate `Var` survives.
    #[test]
    fn collapse_wraps_a_kernel_in_three_seq_folds_around_one_write() {
        let k = Kernel::x().mul(&Kernel::constant(2.0)).add(&Kernel::y());
        let (arena, root) = k.parts();
        let mut arena = arena.clone();
        let domain = Domain {
            shape: LatticeShape::new([5, 3]),
            origin: mint_origin(),
        };
        let new_root = collapse(&mut arena, root, domain);

        let ExprNode::Reduce {
            fold: Fold::Range(row_fold),
            body: col_id,
        } = arena.node(new_root)
        else {
            panic!("collapse's root must be a Reduce");
        };
        assert_eq!(row_fold.monoid(), Monoid::SEQ);
        assert_eq!(row_fold.range(), 0..3);

        let ExprNode::Reduce {
            fold: Fold::Range(col_fold),
            body: lane_id,
        } = arena.node(col_id)
        else {
            panic!("the row fold's body must be a Reduce");
        };
        assert_eq!(col_fold.monoid(), Monoid::SEQ);
        assert_eq!(col_fold.range(), 0..5);

        let ExprNode::Reduce {
            fold: Fold::Range(lane_fold),
            body: write,
        } = arena.node(lane_id)
        else {
            panic!("the col fold's body must be a Reduce");
        };
        assert_eq!(lane_fold.monoid(), Monoid::SEQ);
        assert_eq!(lane_fold.range(), 0..1);

        let ExprNode::Write {
            row,
            col,
            lane,
            value,
        } = arena.node(write)
        else {
            panic!("the lane fold's body must be a Write");
        };
        assert_eq!(
            (row, col, lane),
            (row_fold.binder(), col_fold.binder(), lane_fold.binder()),
            "the Write's binders must be the three folds' own, in order"
        );

        assert_eq!(arena.uniforms().len(), 2, "x0 and y0, and nothing else");
        assert!(
            !reachable_var(&arena, new_root, 0),
            "no X survives collapse"
        );
        assert!(
            !reachable_var(&arena, new_root, 1),
            "no Y survives collapse"
        );

        let x0_pos = arena
            .uniforms()
            .iter()
            .position(|d| d.id == domain.origin[0].id)
            .expect("x0 declared");
        let y0_pos = arena
            .uniforms()
            .iter()
            .position(|d| d.id == domain.origin[1].id)
            .expect("y0 declared");
        let expected_x = alloc::format!(
            "add(add(Uniform({x0_pos}), Var({})), Var({}))",
            col_fold.binder().var(),
            lane_fold.binder().var()
        );
        let expected_y = alloc::format!("add(Uniform({y0_pos}), Var({}))", row_fold.binder().var());
        let shown = alloc::format!("{}", arena.display(value));
        assert!(shown.contains(&expected_x), "shown: {shown}");
        assert!(shown.contains(&expected_y), "shown: {shown}");
    }

    /// `sum_over(4, |i| X + i)` binds slot 0 for its own fold before
    /// `collapse` ever runs, so the lattice must take 1, 2, 3 — and the
    /// kernel's own fold must still read its own binder afterward: no
    /// capture, mirroring `variance.rs`'s
    /// `six_nested_binders_each_take_a_slot`.
    #[test]
    fn collapse_takes_the_slots_the_kernel_leaves_free() {
        let k = Kernel::sum_over(4, |i| Kernel::x().add(i));
        let (arena, root) = k.parts();
        let mut arena = arena.clone();

        let ExprNode::Reduce { fold, .. } = arena.node(root) else {
            panic!("sum_over must build a Reduce root");
        };
        assert_eq!(
            fold.binder().slot(),
            0,
            "test setup: the kernel's own fold binds slot 0"
        );

        let domain = Domain {
            shape: LatticeShape::new([2, 2]),
            origin: mint_origin(),
        };
        let new_root = collapse(&mut arena, root, domain);

        let (row_fold, col_fold, lane_fold, write) = collapse_shape(&arena, new_root);
        assert_eq!(
            [
                row_fold.binder().slot(),
                col_fold.binder().slot(),
                lane_fold.binder().slot()
            ],
            [1, 2, 3],
            "slot 0 is taken, so the lattice takes 1, 2, 3 in ascending order"
        );

        let ExprNode::Write { value, .. } = arena.node(write) else {
            panic!("collapse must build a Write");
        };
        assert!(
            reachable_var(&arena, value, Binder::from_slot(0).expect("slot 0").var()),
            "the kernel's own fold must still read its own binder after collapse"
        );
    }

    /// After collapsing `X + Y`, every bit is accounted for: the root is
    /// fully bound, the `Write` and the value it stores depend on exactly
    /// the three binders — via `x`/`y` for the value — and neither depends
    /// on a coordinate, because none survives.
    #[test]
    fn every_binder_is_bound_at_the_root_and_read_by_the_write() {
        let k = Kernel::x().add(&Kernel::y());
        let (arena, root) = k.parts();
        let mut arena = arena.clone();
        let domain = Domain {
            shape: LatticeShape::new([4, 4]),
            origin: mint_origin(),
        };
        let new_root = collapse(&mut arena, root, domain);

        let (row_fold, col_fold, lane_fold, write) = collapse_shape(&arena, new_root);
        let ExprNode::Write { value, .. } = arena.node(write) else {
            panic!("collapse must build a Write");
        };

        let v = compute_arena_variance(&arena);
        assert_eq!(
            v[new_root.0 as usize],
            Variance::CONST,
            "every binder is bound by the time the root is reached"
        );

        let all_three = Variance::from_var(row_fold.binder().var())
            .union(Variance::from_var(col_fold.binder().var()))
            .union(Variance::from_var(lane_fold.binder().var()));
        assert_eq!(
            v[write.0 as usize], all_three,
            "the store reads all three binders for its address"
        );
        assert_eq!(
            v[value.0 as usize], all_three,
            "the value depends on all three, via x and y, and nothing else"
        );
        assert!(
            !v[value.0 as usize].is_spatially_varying(),
            "no coordinate bit survives collapse"
        );
    }

    /// A lane-uniform read stays lane-uniform through collapse: `Gather(buf,
    /// 0.5, Y)`'s row-only address carries only the row bit; `Gather(buf, X,
    /// Y)` carries all three. This is the bit step 6 reads to choose a
    /// broadcast load over a gather.
    #[test]
    fn a_row_only_read_stays_lane_uniform_through_collapse() {
        let mut arena = ExprArena::new();
        let buf_id = arena.declare_buffer(BufferDecl {
            id: BufferIdentity::mint(),
            width: 8,
            height: 8,
        });

        let buf1 = arena.push_buffer(buf_id);
        let half = arena.push_const(0.5);
        let y1 = arena.push_var(1);
        let row_only = arena.push_ternary(OpKind::Gather, buf1, half, y1);

        let buf2 = arena.push_buffer(buf_id);
        let x2 = arena.push_var(0);
        let y2 = arena.push_var(1);
        let all_three_read = arena.push_ternary(OpKind::Gather, buf2, x2, y2);

        let root = arena.push_binary(OpKind::Add, row_only, all_three_read);

        let domain = Domain {
            shape: LatticeShape::new([4, 4]),
            origin: mint_origin(),
        };
        let new_root = collapse(&mut arena, root, domain);

        let (row_fold, col_fold, lane_fold, write) = collapse_shape(&arena, new_root);
        let ExprNode::Write { value, .. } = arena.node(write) else {
            panic!("collapse must build a Write");
        };
        let ExprNode::Binary(OpKind::Add, new_row_only, new_all_three) = arena.node(value) else {
            panic!("the value must still be Add(row_only, all_three_read)");
        };

        let v = compute_arena_variance(&arena);
        let (row, col, lane) = (
            row_fold.binder().var(),
            col_fold.binder().var(),
            lane_fold.binder().var(),
        );

        assert!(v[new_row_only.0 as usize].depends_on(row));
        assert!(v[new_row_only.0 as usize].is_invariant_in(col));
        assert!(v[new_row_only.0 as usize].is_invariant_in(lane));

        assert!(v[new_all_three.0 as usize].depends_on(row));
        assert!(v[new_all_three.0 as usize].depends_on(col));
        assert!(v[new_all_three.0 as usize].depends_on(lane));
    }

    /// `w = 8`, `L = 4`: an exact multiple, so `pack` builds one strided fold
    /// and no remainder — no `Seq` anywhere reachable, and the `Write` is
    /// shared with what `collapse` built, not copied.
    #[test]
    fn pack_strip_mines_an_exact_multiple_with_no_remainder() {
        let k = Kernel::x().add(&Kernel::y());
        let (arena, root) = k.parts();
        let mut arena = arena.clone();
        let domain = Domain {
            shape: LatticeShape::new([8, 3]),
            origin: mint_origin(),
        };
        let collapsed = collapse(&mut arena, root, domain);
        let (_, _, _, write_before) = collapse_shape(&arena, collapsed);

        let packed = pack(&mut arena, collapsed, 4);

        let ExprNode::Reduce {
            fold: Fold::Range(row_fold),
            body: row_body,
        } = arena.node(packed)
        else {
            panic!("packed root must be the row fold");
        };
        assert_eq!(row_fold.monoid(), Monoid::SEQ);
        assert_eq!(row_fold.range(), 0..3);

        let ExprNode::Reduce {
            fold: Fold::Range(col_fold),
            body: lane_id,
        } = arena.node(row_body)
        else {
            panic!("the row fold's body must be the strided col fold");
        };
        assert_eq!(col_fold.range(), 0..8);
        assert_eq!(col_fold.stride(), 4);
        assert_eq!(col_fold.len(), 2);

        let ExprNode::Reduce {
            fold: Fold::Range(lane_fold),
            body: write,
        } = arena.node(lane_id)
        else {
            panic!("the col fold's body must be the lane fold");
        };
        assert_eq!(lane_fold.range(), 0..4);
        assert_eq!(lane_fold.stride(), 1);

        assert_eq!(write, write_before, "pack shares collapse's Write");
        assert!(!reachable_seq(&arena, packed), "no remainder means no Seq");
    }

    /// `w = 10`, `L = 4`: `r = 2`, so `pack` sequences a main fold over
    /// `[0,8)` and a remainder over `[8,9) × [0,2)` — both lane folds sharing
    /// the one `Write`, and the whole tree still fully bound.
    #[test]
    fn pack_sequences_the_remainder_after_the_main_fold() {
        let k = Kernel::x().add(&Kernel::y());
        let (arena, root) = k.parts();
        let mut arena = arena.clone();
        let domain = Domain {
            shape: LatticeShape::new([10, 2]),
            origin: mint_origin(),
        };
        let collapsed = collapse(&mut arena, root, domain);
        let (_, _, _, write_before) = collapse_shape(&arena, collapsed);

        let packed = pack(&mut arena, collapsed, 4);

        let ExprNode::Reduce {
            fold: Fold::Range(row_fold),
            body: row_body,
        } = arena.node(packed)
        else {
            panic!("packed root must be the row fold");
        };
        assert_eq!(row_fold.range(), 0..2);

        let ExprNode::Binary(OpKind::Seq, main, rem) = arena.node(row_body) else {
            panic!("a nonzero remainder must sequence main then remainder");
        };

        let ExprNode::Reduce {
            fold: Fold::Range(main_col),
            body: main_lane_id,
        } = arena.node(main)
        else {
            panic!("main must be the strided col fold");
        };
        assert_eq!(main_col.range(), 0..8);
        assert_eq!(main_col.stride(), 4);
        let ExprNode::Reduce {
            fold: Fold::Range(main_lane),
            body: main_write,
        } = arena.node(main_lane_id)
        else {
            panic!("main's body must be the lane fold");
        };
        assert_eq!(main_lane.range(), 0..4);

        let ExprNode::Reduce {
            fold: Fold::Range(rem_col),
            body: rem_lane_id,
        } = arena.node(rem)
        else {
            panic!("rem must be the [8,9) col fold");
        };
        assert_eq!(rem_col.range(), 8..9);
        assert_eq!(rem_col.stride(), 1);
        let ExprNode::Reduce {
            fold: Fold::Range(rem_lane),
            body: rem_write,
        } = arena.node(rem_lane_id)
        else {
            panic!("rem's body must be the lane fold");
        };
        assert_eq!(rem_lane.range(), 0..2);

        assert_eq!(main_write, write_before);
        assert_eq!(
            rem_write, write_before,
            "both lane folds' body is the same Write"
        );

        let v = compute_arena_variance(&arena);
        assert_eq!(v[packed.0 as usize], Variance::CONST);
    }

    /// `w = 3`, `L = 4`: a row narrower than one batch. The main fold is the
    /// empty strided fold `[0,0)` — no special case — and the whole row is
    /// the remainder, `[0,1) × [0,3)`.
    #[test]
    fn pack_of_a_row_narrower_than_a_batch_is_all_remainder() {
        let k = Kernel::x().add(&Kernel::y());
        let (arena, root) = k.parts();
        let mut arena = arena.clone();
        let domain = Domain {
            shape: LatticeShape::new([3, 1]),
            origin: mint_origin(),
        };
        let collapsed = collapse(&mut arena, root, domain);

        let packed = pack(&mut arena, collapsed, 4);

        let ExprNode::Reduce { body: row_body, .. } = arena.node(packed) else {
            panic!("packed root must be the row fold");
        };

        let ExprNode::Binary(OpKind::Seq, main, rem) = arena.node(row_body) else {
            panic!("even an all-remainder row sequences main then remainder");
        };

        let ExprNode::Reduce {
            fold: Fold::Range(main_col),
            ..
        } = arena.node(main)
        else {
            panic!("main must be a Reduce");
        };
        assert!(main_col.is_empty());
        assert_eq!(main_col.range(), 0..0);
        assert_eq!(main_col.stride(), 4);

        let ExprNode::Reduce {
            fold: Fold::Range(rem_col),
            body: rem_lane_id,
        } = arena.node(rem)
        else {
            panic!("rem must be a Reduce");
        };
        assert_eq!(rem_col.range(), 0..1);
        let ExprNode::Reduce {
            fold: Fold::Range(rem_lane),
            ..
        } = arena.node(rem_lane_id)
        else {
            panic!("rem's body must be the lane fold");
        };
        assert_eq!(rem_lane.range(), 0..3);
    }

    /// `pack` on anything but `collapse`'s own output is a pipeline-order
    /// bug, refused by shape rather than guessed at.
    #[test]
    #[should_panic(expected = "pack expects collapse's shape")]
    fn pack_refuses_a_foreign_root() {
        let k = Kernel::x().add(&Kernel::y());
        let (arena, root) = k.parts();
        let mut arena = arena.clone();
        let _ = pack(&mut arena, root, 4);
    }

    /// A `Ref` reaching `collapse` is a name `substitute_vars_with` would
    /// carry through unresolved — refused rather than silently wrong.
    #[test]
    #[should_panic(expected = "is a Ref reachable from root")]
    fn collapse_refuses_a_ref() {
        let named = Kernel::x();
        let (named_arena, named_root) = named.parts();
        let key = KernelKey::of(named_arena, named_root);

        let mut arena = ExprArena::new();
        let r = arena.push_ref(key);
        let root = arena.push_binary(OpKind::Add, r, r);

        let domain = Domain {
            shape: LatticeShape::new([2, 2]),
            origin: mint_origin(),
        };
        let _ = collapse(&mut arena, root, domain);
    }

    /// A `Dwrt` reaching `collapse` has not been taken with respect to
    /// `Var(0)` yet — refused rather than substituted with the wrong
    /// chain-rule factor.
    #[test]
    #[should_panic(expected = "a Dwrt is reachable from root")]
    fn collapse_refuses_a_dwrt() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let wrt = arena.push_const(0.0);
        let root = arena.push_binary(OpKind::Dwrt, x, wrt);

        let domain = Domain {
            shape: LatticeShape::new([2, 2]),
            origin: mint_origin(),
        };
        let _ = collapse(&mut arena, root, domain);
    }

    /// An integral is not a loop, so there is nothing for the lattice to
    /// wrap: `legalize` replaces every one by its quadrature before this
    /// pass, and one reaching it is a caller that skipped `resolve`.
    #[test]
    #[should_panic(expected = "an interval fold")]
    fn collapse_refuses_an_interval() {
        let area = Kernel::x().area();
        let (arena, root) = area.parts();
        let mut arena = arena.clone();
        let domain = Domain {
            shape: LatticeShape::new([2, 2]),
            origin: mint_origin(),
        };
        let _ = collapse(&mut arena, root, domain);
    }
}
