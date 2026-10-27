//! The CPU's factorization of the lattice: rows, batches and lanes.
//!
//! The lattice is the kernel's domain; how a backend walks it is the
//! backend's schedule (docs/language/2026-10-15-kernels-evaluate-kernels.md §3,
//! docs/codegen/2026-10-15-a-schedule-is-a-factorization.md). The CPU's walk
//! is a [`Traversal`] — three binders, an extent, and the tier's lane width —
//! and this module holds no arena node for it: [`Collapse`] only spells the
//! kernel's coordinates through the traversal's binders, and `Lower` writes
//! the loops themselves into the schedule ([`Traversal::pieces`]).
//!
//! ```text
//! collapse(f) = f(x0 + i + l, y0 + j)             i, j, l the traversal's binders
//! walk        = fold_{j∈[0,h)} (   fold_{i∈[0,w−r) step L} store L lanes
//!                                ; fold_{i∈[w−r,w−r+1)}    store r lanes )
//!                 (the first arm only when w − r > 0, the second only when r > 0,
//!                  the `;` only when both exist)
//! ```
//!
//! `w × h` is the extent, `L` the lanes, `r = w % L`. This is the monadic
//! reading of the chunking law
//! `for_ [0,w) f = for_ [0,w−r) step L (\i -> for_ [0,L) (\l -> f(i+l))) >>
//! for_ [w−r] (\i -> for_ [0,r) (\l -> f(i+l)))`
//! (docs/archive/plans/2026-09-16-collapse-is-a-fold.md §2.1–§2.4).

use crate::error::CompileError;
use crate::pass::Pass;
use crate::program::Looping;
use pixelflow_ir::LatticeShape;
use pixelflow_ir::arena::{COORD_AXES, ExprArena, ExprId, ExprNode, UniformDecl, UniformId};
use pixelflow_ir::fold::Binder;
use pixelflow_ir::kind::OpKind;
use pixelflow_ir::variance::Variance;

/// The domain a collapse tabulates a kernel over: its extent, and the origin
/// the call supplies. `LatticeShape` is "the compile-time half of
/// `pixelflow_core::Lattice`, origin erased"; this puts the origin back as
/// two uniforms.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Domain {
    /// Samples per axis, `[w, h]`: the two outer folds' trip counts.
    pub(crate) shape: LatticeShape,
    /// `[x0, y0]`: the lattice position of sample `(0, 0)`, bound per call.
    pub(crate) origin: [UniformDecl; COORD_AXES],
}

/// How the CPU walks a lattice: the binders its row, column and lane loops
/// count with, the extent, and the lanes per batch.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Traversal {
    /// The row loop's binder.
    pub(crate) row: Binder,
    /// The column loop's binder.
    pub(crate) col: Binder,
    /// The lane binder: executed by lanes, never a loop counter.
    pub(crate) lane: Binder,
    /// Samples per axis, `[w, h]`.
    pub(crate) extent: [u32; COORD_AXES],
    /// Lanes per batch: the tier's width in `f32`s.
    pub(crate) lanes: u32,
}

/// One store loop of a [`Traversal`]'s row: the column fold's binder, stride
/// and range, and the lanes it stores per trip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Piece {
    /// The column loop: strided by the lane width for full batches.
    pub(crate) col: Looping,
    /// The column fold's range, `[lo, hi)`.
    pub(crate) range: [u64; 2],
    /// Lanes stored per trip: the batch, or the row's remainder.
    pub(crate) lanes: u32,
}

impl Traversal {
    /// The variance of the traversal's three binders together: what a value
    /// reads when it is read inside all three loops.
    pub(crate) fn variance(&self) -> Variance {
        [self.row, self.col, self.lane]
            .into_iter()
            .fold(Variance::CONST, |v, binder| {
                v.union(Variance::from_var(binder.var()))
            })
    }

    /// The column loops a row is, in order: the full batches, then the
    /// remainder. A loop exists only when it has trips: the batches when
    /// `w − r > 0`, the remainder when `r > 0`. When `w < L` there is no full
    /// batch, so the whole column is the remainder.
    ///
    /// # Panics
    ///
    /// If the lattice has no column, or no lane.
    pub(crate) fn pieces(&self) -> Vec<Piece> {
        assert!(self.lanes != 0, "a traversal needs a nonzero lane count");
        let width = u64::from(self.extent[0]);
        let lanes = u64::from(self.lanes);
        let remainder = width % lanes;
        let batches = (width > remainder).then(|| Piece {
            col: Looping::Sweep {
                binder: self.col,
                stride: lanes,
            },
            range: [0, width - remainder],
            lanes: self.lanes,
        });
        let tail = (remainder > 0).then(|| Piece {
            col: Looping::Sweep {
                binder: self.col,
                stride: 1,
            },
            range: [width - remainder, width - remainder + 1],
            lanes: remainder as u32,
        });
        let pieces: Vec<Piece> = batches.into_iter().chain(tail).collect();
        assert!(!pieces.is_empty(), "the lattice has no column");
        pieces
    }
}

/// A legalized kernel with the traversal that walks it: the arena holds only
/// the kernel, spelled through the traversal's binders.
pub(crate) struct Factored {
    pub(crate) arena: ExprArena,
    pub(crate) root: ExprId,
    pub(crate) traversal: Traversal,
}

/// [`collapse`] as a stage: choose the traversal's binders and spell the
/// kernel's coordinates through them. `lanes` is the width of the tier the
/// process chose.
pub(crate) struct Collapse {
    /// The extent and the origin.
    pub(crate) domain: Domain,
    /// Lanes per batch.
    pub(crate) lanes: u32,
}

impl Pass for Collapse {
    type Input = (ExprArena, ExprId);
    type Output = Factored;

    fn run(self, (mut arena, root): Self::Input) -> Result<Self::Output, CompileError> {
        assert!(self.lanes != 0, "collapse: lanes must be nonzero");
        let (root, traversal) = collapse(&mut arena, root, self.domain, self.lanes);
        Ok(Factored {
            arena,
            root,
            traversal,
        })
    }
}

/// Spell the kernel at `root` through a traversal's three binders and return
/// the new root with the traversal. No loop is built here; `Lower` writes them.
///
/// `f ↦ f.at(x0+i+l, y0+j)`. `x0`/`y0` are `domain.origin`, declared as
/// uniforms; the warped `x` is built `Add(Add(Uniform(x0), Var(col)),
/// Var(lane))` — `(x0 + col)` is lane-uniform and hoists to the batch scope —
/// and `y` is `Add(Uniform(y0), Var(row))`. Building the warp this way and
/// substituting it for `Var(0)`/`Var(1)` is exactly `Kernel::at`.
///
/// # Panics
///
/// - If a `Ref` or a `Dwrt` (binary or n-ary) is reachable from
///   `root`. A `Ref` is a name:
///   [`substitute_vars_with`](ExprArena::substitute_vars_with) copies it
///   through without reaching the referent's `Var(0)`, so the warp below
///   would silently never reach it. A `Dwrt` must be taken with respect to
///   `Var(0)` before this pass substitutes that variable away, or the
///   chain-rule factor it needs is gone. Run `expand_refs`, then
///   `lower_dwrt`, first — see `pipeline.rs` for the order.
/// - If fewer than three binder slots are free — neither bound by a
///   reachable `Reduce` nor read by a reachable `Var` in the binder range: a
///   kernel already nesting three folds of its own leaves nothing for the
///   lattice.
fn collapse(
    arena: &mut ExprArena,
    root: ExprId,
    domain: Domain,
    lanes: u32,
) -> (ExprId, Traversal) {
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

    let [x0_id, y0_id] = domain.origin.map(|decl| origin_slot(arena, decl));
    let x0 = arena.push_uniform(x0_id);
    let y0 = arena.push_uniform(y0_id);

    let i = arena.push_var(col.var());
    let l = arena.push_var(lane.var());
    let j = arena.push_var(row.var());

    // `(x0 + i)` first, then `+ l`: the inner sum reads no lane index, so it
    // is lane-uniform and hoists to the batch scope by the same variance bit
    // that chooses a broadcast load over a gather.
    let x0_plus_i = arena.push_binary(OpKind::Add, x0, i);
    let x = arena.push_binary(OpKind::Add, x0_plus_i, l);
    let y = arena.push_binary(OpKind::Add, y0, j);

    let body = arena.substitute_vars_with(root, &[(0, x), (1, y)]);
    let traversal = Traversal {
        row,
        col,
        lane,
        extent: domain.shape.extent(),
        lanes,
    };
    (body, traversal)
}

/// The slot naming `decl`'s instance in `arena`, declared if this is the first
/// time that identity has been seen. Two declarations of one identity are one
/// slot, so a kernel that already names the origin shares it.
fn origin_slot(arena: &mut ExprArena, decl: UniformDecl) -> UniformId {
    match arena.uniforms().iter().position(|d| d.id == decl.id) {
        Some(slot) => UniformId(slot as u64),
        None => arena.declare_uniform(decl),
    }
}

/// The full pass order, named once so every panic message in this file spells
/// it identically.
const PASS_ORDER: &str = "expand_refs -> lower_dwrt -> collapse -> expand_gather \
     -> expand_transcendentals -> guard";

/// Which binder slots [`collapse`] must not choose — bound by a reachable
/// `Reduce`, or read by a reachable `Var` in the binder range — found in one
/// walk of `root`'s reachable subgraph that also refuses, by panicking and
/// naming [`PASS_ORDER`], any reachable `Ref` or `Dwrt` (see
/// [`collapse`]'s own doc for why each is refused).
fn reachable_taken_binders(arena: &ExprArena, root: ExprId) -> [bool; Binder::COUNT] {
    let mut taken = [false; Binder::COUNT];
    let mut seen = vec![false; arena.len()];
    let mut stack = vec![root];
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
            ExprNode::Binary(OpKind::Dwrt, _, _) | ExprNode::Nary(OpKind::Dwrt, _) => panic!(
                "collapse: a Dwrt is reachable from root — it must be taken \
                 with respect to Var(0) before this pass substitutes that \
                 variable away; run lower_dwrt first ({PASS_ORDER})"
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

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::arena::{BufferDecl, BufferIdentity, UniformIdentity};
    use pixelflow_ir::variance::compute_arena_variance;
    use pixelflow_ir::{Kernel, key::KernelKey};

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

    fn domain(extent: [u32; 2]) -> Domain {
        Domain {
            shape: LatticeShape::new(extent),
            origin: mint_origin(),
        }
    }

    /// A traversal of `extent` at `lanes`, for the tests of its pieces.
    fn traversal(extent: [u32; 2], lanes: u32) -> Traversal {
        let slot = |n| Binder::from_slot(n).expect("the slot exists");
        Traversal {
            row: slot(0),
            col: slot(1),
            lane: slot(2),
            extent,
            lanes,
        }
    }

    /// Whether `Var(target)` is reachable from `root` — the check
    /// `variance.rs`'s `six_nested_binders_each_take_a_slot` uses to confirm
    /// no rename captured an index it should not have.
    fn reachable_var(arena: &ExprArena, root: ExprId, target: u8) -> bool {
        let mut seen = vec![false; arena.len()];
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            let idx = id.0 as usize;
            if core::mem::replace(&mut seen[idx], true) {
                continue;
            }
            if matches!(arena.node(id), ExprNode::Var(i) if i == target) {
                return true;
            }
            stack.extend(arena.children(id));
        }
        false
    }

    /// `collapse` builds no loop: the kernel comes back as a value over the
    /// traversal's binders, with no coordinate `Var` left, and the traversal
    /// carries the extent and lanes it was asked for.
    #[test]
    fn collapse_spells_the_coordinates_through_the_binders() {
        let k = Kernel::x().add(&Kernel::y());
        let (arena, root) = k.parts();
        let mut arena = arena.clone();
        let (body, t) = collapse(&mut arena, root, domain([8, 3]), 4);

        assert!(!matches!(arena.node(body), ExprNode::Reduce { .. }));
        assert!(!reachable_var(&arena, body, 0), "X is substituted away");
        assert!(!reachable_var(&arena, body, 1), "Y is substituted away");
        assert_eq!((t.extent, t.lanes), ([8, 3], 4));

        let v = compute_arena_variance(&arena)[body.0 as usize];
        assert_eq!(
            v,
            t.variance(),
            "the value depends on all three binders, via x and y, and nothing else"
        );
    }

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

        let (body, t) = collapse(&mut arena, root, domain([2, 2]), 4);
        assert_eq!(
            [t.row.slot(), t.col.slot(), t.lane.slot()],
            [1, 2, 3],
            "slot 0 is taken, so the lattice takes 1, 2, 3 in ascending order"
        );
        assert!(
            reachable_var(&arena, body, Binder::from_slot(0).expect("slot 0").var()),
            "the kernel's own fold must still read its own binder after collapse"
        );
    }

    /// A lane-uniform read stays lane-uniform through collapse: `Gather(buf,
    /// 0.5, Y)`'s row-only address carries only the row bit; `Gather(buf, X,
    /// Y)` carries all three. This is the bit lowering reads to choose a
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
        let (body, t) = collapse(&mut arena, root, domain([4, 4]), 4);

        let v = compute_arena_variance(&arena);
        let ExprNode::Binary(OpKind::Add, row_read, full_read) = arena.node(body) else {
            panic!("the sum survives collapse");
        };
        assert_eq!(v[row_read.0 as usize], Variance::from_var(t.row.var()));
        assert_eq!(v[full_read.0 as usize], t.variance());
    }

    /// An exact multiple of the lanes is all full batches: no remainder.
    #[test]
    fn an_exact_multiple_has_no_remainder() {
        let t = traversal([8, 2], 4);
        let pieces = t.pieces();
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].range, [0, 8]);
        assert_eq!(pieces[0].lanes, 4);
        assert_eq!(
            pieces[0].col,
            Looping::Sweep {
                binder: t.col,
                stride: 4
            }
        );
    }

    /// The remainder comes after the batches, one trip of `r` lanes at `w − r`.
    #[test]
    fn the_remainder_follows_the_batches() {
        let pieces = traversal([10, 1], 4).pieces();
        assert_eq!(pieces.len(), 2);
        assert_eq!((pieces[0].range, pieces[0].lanes), ([0, 8], 4));
        assert_eq!((pieces[1].range, pieces[1].lanes), ([8, 9], 2));
    }

    /// A row narrower than a batch is all remainder.
    #[test]
    fn a_row_narrower_than_a_batch_is_all_remainder() {
        let pieces = traversal([3, 1], 4).pieces();
        assert_eq!(pieces.len(), 1);
        assert_eq!((pieces[0].range, pieces[0].lanes), ([0, 1], 3));
    }

    #[test]
    #[should_panic(expected = "is a Ref reachable from root")]
    fn collapse_refuses_a_ref() {
        let named = Kernel::x();
        let (named_arena, named_root) = named.parts();
        let key = KernelKey::of(named_arena, named_root);

        let mut arena = ExprArena::new();
        let r = arena.push_ref(key);
        let root = arena.push_binary(OpKind::Add, r, r);
        let _ = collapse(&mut arena, root, domain([2, 2]), 4);
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
        let _ = collapse(&mut arena, root, domain([2, 2]), 4);
    }
}
