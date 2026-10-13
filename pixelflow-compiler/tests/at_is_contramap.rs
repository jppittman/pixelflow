//! `f.at(x, y)` is the field `f` observed at `(x, y)`: contramap,
//! `⟦f.at(u, v)⟧(x, y) = ⟦f⟧(⟦u⟧(x, y), ⟦v⟧(x, y))`.
//!
//! Every expression in a body is a field over the two axes, so every `f32`
//! or `bool` may be observed elsewhere — a `let`, a helper's parameter, a
//! kernel's application, a fold. Before this, only a *function* could be:
//! a helper taking its coordinates as arguments, or a kernel-typed
//! parameter. A field already bound to a name had to be rewritten as one
//! first. Lowering builds no node for `.at`; it rewrites its receiver's,
//! through `ExprArena::warp`, the IR's one arena-level definition of the
//! warp, which a kernel's application goes through too.
//!
//! What this file pins:
//!
//! - **the meaning** — the neighbour of a `let`-bound field, by value; the
//!   two substitutions simultaneous; a mask still a mask;
//! - **the laws** — `f.at(X, Y)` is `f`, and `f.at(g).at(h)` is
//!   `f.at(g.at(h))`, each one program by canonical key;
//! - **one program with the builder** — the syntax builds what
//!   `Kernel::at` builds;
//! - **capture avoidance** — a coordinate reading a fold's index, observed
//!   through a field holding its own fold, keeps both binders apart;
//! - **a name is expanded first** — a kernel argument passed `by_ref` and
//!   observed elsewhere is observed there, not at the sample;
//! - **a derivative is of the warped field** — the chain rule, as
//!   `Kernel::at` has it (`derivative_under_warp.rs`), and pointwise
//!   wherever the warp is a translation.

use pixelflow_compiler::{kernel, kernel_raw};
use pixelflow_core::{Kernel, Lattice};
use pixelflow_ir::key::canonical;

/// The sample every value is read at: `X = 3`, `Y = 5`.
const AT: (f32, f32) = (3.0, 5.0);

fn bake(k: &Kernel) -> f32 {
    Lattice::eval_at(k, AT.0, AT.1)
}

/// Whether two kernels are one program: the same canonical key, what the
/// JIT cache keys on.
fn assert_same_program(written: &Kernel, built: &Kernel) {
    let (written_arena, written_root) = written.parts();
    let (built_arena, built_root) = built.parts();
    assert_eq!(
        canonical(written_arena, written_root).key,
        canonical(built_arena, built_root).key,
        "written: {}\nbuilt:   {}",
        written_arena.display(written_root),
        built_arena.display(built_root),
    );
}

fn x() -> Kernel {
    Kernel::x()
}

fn y() -> Kernel {
    Kernel::y()
}

fn constant(v: f32) -> Kernel {
    Kernel::constant(v)
}

kernel! {
    /// `d` one column east of the screen point `(x, y)` — a helper observing
    /// a field it was handed. Its coordinates are arguments, as a helper's
    /// always are; `.at` is in screen space, as every coordinate is.
    fn east(d: f32, x: f32, y: f32) -> f32 { d.at(x + 1.0, y) }

    /// The helper above, handed `X²` and the sample: `(X + 1)²`.
    pub fn east_of_square() -> f32 { east(X * X, X, Y) }

    /// What `k` makes, bound to a name as a field and observed one column
    /// east of the sample. Lowered when the host function is called, so
    /// this is the staged site's `.at`.
    pub fn east_of(k: impl Fn(f32, f32) -> f32) -> f32 {
        let here = k(X, Y);
        here.at(X + 1.0, Y)
    }
}

// ───────────────────────────────── meaning ─────────────────────────────────

/// The case `.at` exists for: a field already bound to a name, observed at
/// the pixel next to this one. Recomputed there, not read from a buffer —
/// it is the same field, elsewhere.
#[test]
fn a_let_bound_field_is_observed_at_its_neighbour() {
    let neighbour = kernel!(|| {
        let d = X * X + Y;
        d.at(X + 1.0, Y) - d
    });
    // (4² + 5) − (3² + 5).
    assert_eq!(bake(&neighbour), 7.0);
}

/// Both coordinates are substituted at once. In sequence, `X := Y` first
/// makes `X − Y` into `Y − Y`, and the second substitution has nothing
/// left to swap.
#[test]
fn the_two_coordinates_are_substituted_simultaneously() {
    let swapped = kernel_raw!(|| (X - Y).at(Y, X));
    assert_eq!(bake(&swapped), 2.0, "5 − 3, not 0");
}

/// A mask observed elsewhere is still a mask, and an `if` chooses by it.
#[test]
fn a_mask_observed_elsewhere_is_still_a_mask() {
    let chosen = kernel!(|| if (X < Y).at(Y, X) { 1.0 } else { 0.0 });
    assert_eq!(bake(&chosen), 0.0, "5 < 3 is false");
}

/// A helper may observe a field it was handed: the coordinates it passes
/// are screen coordinates, so it reaches the same point an entry would.
#[test]
fn a_helper_observes_a_field_it_was_handed() {
    assert_eq!(bake(&east_of_square()), 16.0, "(3 + 1)²");
}

// ───────────────────────────────── the laws ─────────────────────────────────

/// Identity: a field observed at the sample is the field. Not merely the
/// same value — the same program, so no warp at `(X, Y)` costs a cache
/// entry of its own.
#[test]
fn a_field_at_the_sample_is_the_field() {
    assert_same_program(
        &kernel_raw!(|| (X * X + Y).at(X, Y)),
        &kernel_raw!(|| X * X + Y),
    );
}

/// Composition: observing at `g` and then at `h` is observing at `g`
/// observed at `h` — `(f ∘ g) ∘ h = f ∘ (g ∘ h)`, with the warps read as the
/// maps they are. Here `g = (X + 1, Y)` and `h = (2X, Y)`, so both sides are
/// `f` at `(2X + 1, Y)`.
#[test]
fn observing_twice_is_observing_at_the_composite() {
    let twice = kernel_raw!(|| (X * X + Y).at(X + 1.0, Y).at(X * 2.0, Y));
    let composite = kernel_raw!(|| (X * X + Y).at((X + 1.0).at(X * 2.0, Y), Y));
    assert_same_program(&twice, &composite);
    assert_eq!(bake(&twice), 54.0, "(2·3 + 1)² + 5");
}

// ──────────────────────────── one program, two spellings ────────────────────

/// The syntax builds what the builder builds: `.at` in a body and
/// `Kernel::at` on the same field and coordinates are one program, so a
/// warp written either way shares a JIT cache entry.
#[test]
fn at_in_the_syntax_is_kernel_at() {
    let written = kernel_raw!(|| (X * X + Y).at(X + 1.0, Y * 2.0));
    let built = kernel_raw!(|| X * X + Y).at(&x().add(&constant(1.0)), &y().mul(&constant(2.0)));
    assert_same_program(&written, &built);
    assert_eq!(bake(&written), 26.0, "(3 + 1)² + 2·5");
}

// ──────────────────────────── binders and names ─────────────────────────────

/// A coordinate that reads a fold's index, observing a field that holds a
/// fold of its own. After the warp the outer index sits under the inner
/// fold; it must still name the outer fold. The outer fold's slot is chosen
/// after its body exists, as the lowest the body leaves free, so it cannot
/// be the inner one's.
///
/// Captured, the outer index would read the inner one's: `Σᵢ Σⱼ (X + 2j)`,
/// independent of `i`, which is 30 here — plausible, and wrong.
#[test]
fn a_fold_index_in_a_coordinate_is_not_captured_by_the_fields_fold() {
    let written = kernel_raw!(|| {
        let f = (0..3).map(|j| X + j as f32).sum();
        (0..2).map(|i| f.at(X + i as f32, Y)).sum()
    });
    // Σᵢ f(X + i) with f = 3X + 3: (3·3 + 3) + (3·4 + 3).
    assert_eq!(bake(&written), 27.0);

    let field = Kernel::sum_over(3, |j| x().add(j));
    let built = Kernel::sum_over(2, |i| field.at(&x().add(i), &y()));
    assert_same_program(&written, &built);
}

/// A kernel argument passed by name is a `Ref`, and a substitution cannot
/// reach inside a name: left in place, it would be observed at the sample
/// instead of where it was asked to be — 30 here, not 40. The warp expands
/// it first, so the name and the value it names are observed at the same
/// point.
#[test]
fn a_named_kernel_is_observed_where_it_is_asked_to_be() {
    let k = x().mul(&constant(10.0));
    assert_eq!(bake(&east_of(&k)), 40.0, "10 · (3 + 1)");
    assert_eq!(bake(&east_of(&k.by_ref())), 40.0, "the name, expanded");
}

// ─────────────────────────────── derivatives ────────────────────────────────

/// Under a scaling, a derivative observed elsewhere is the derivative of
/// the field observed there: `DX(X²)` at `(2X, Y)` is `d/dX (2X)² = 8X`,
/// not `2·(2X)`. It is how `Kernel::at` has always read one, and what keeps
/// a glyph's antialiasing ramp a screen pixel wide at any scale; this is
/// that contract in the syntax.
#[test]
fn a_derivative_observed_under_a_scaling_follows_the_chain_rule() {
    const CHAIN_RULE_AT_3: f32 = 24.0;
    assert_eq!(
        bake(&kernel_raw!(|| DX(X * X).at(X * 2.0, Y))),
        CHAIN_RULE_AT_3
    );
    assert_eq!(bake(&kernel!(|| DX(X * X).at(X * 2.0, Y))), CHAIN_RULE_AT_3);
}

/// Under a translation the two readings agree — its Jacobian is the
/// identity — so the slope of a field at the neighbouring pixel is what
/// it looks like: `DX(X²)` one column east is `2·(X + 1)`.
#[test]
fn a_derivative_observed_at_a_neighbour_is_the_neighbours_slope() {
    assert_eq!(bake(&kernel!(|| DX(X * X).at(X + 1.0, Y))), 8.0);
}
