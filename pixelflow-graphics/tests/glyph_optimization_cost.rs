//! **What a glyph costs the compiler, bounded — at the two places where
//! "cost" means different things.**
//!
//! Legalization (`LowerDwrt`, `ExpandReduce`) is the *last* pass and a
//! fallback: it takes whatever illegal shape survived saturation and makes it
//! emittable. It owns nothing the e-graph does not also know — the chain rule
//! and a fold's decompositions are rule sets — so running it earlier only
//! takes choices away.
//!
//! That asymmetry is the whole reason this file has two tests instead of one.
//! **Size before saturation is dangerous; size after it is not.** An e-graph
//! is superlinear in what it is handed, so unrolling a 34-piece fold *first*
//! means saturating 141,530 nodes for one glyph and spending the entire
//! budget on it. The assembler is linear and does not care — a million-node
//! IR is a routine afternoon for a register allocator. So the number to hold
//! down is the one the e-graph is fed, and the emitted count is expected to
//! rise when the legalizer moves to the end.
//!
//! [`the_egraph_is_fed_a_program_it_can_reason_about`] is therefore the gate
//! that matters, and it is a *scaling* claim rather than a ceiling: a glyph's
//! program must not grow with its piece count. [`a_glyph_costs_no_more_than
//! _it_did`] is the looser one — emitted size still bounds I-cache and the
//! assembler's input, and a change that silently *doubles* it is still worth
//! catching, even though shaving it is not the objective.
//!
//! **Ceilings, not pins.** Saturation is deterministic — budgets are
//! deterministic functions of the input, so two machines cannot disagree
//! (CLAUDE.md, "A kernel built differently on two machines?") — and an exact
//! pin would therefore be legitimate. It would also have to be edited by
//! every change that *improves* the number, which is how a pin stops being
//! read and starts being rubber-stamped. A ceiling with ~10% headroom passes
//! silently when the compiler gets better and fails when it gets a fifth
//! worse. If one of these fails low by a wide margin, lower it — that is the
//! ratchet, done deliberately.

use pixelflow_graphics::fonts::Font;
use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_ir::LatticeShape;

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// `(character, tile px, ceiling)`. Measured 2026-09-09 with the legalizer
/// last, plus ~10%: `A` 1457, `O` 3633, `8` 8241.
///
/// These are ~22–39% above the counts the same glyphs emitted when
/// `ExpandReduce` ran *before* saturation, and that is the trade named in the
/// module docs, not a regression: the e-graph stopped being handed the
/// unrolled program (141,530 nodes for `8`, now 2,881) and pays for it in
/// emitted size, which is the cheap side.
///
/// Raised again 2026-09-16 (`A` 1446 → 2076, `O` 3588 → 4092, `8` unchanged
/// at 8124) for the same reason under a new name: bucketed trip counts
/// (`docs/plans/2026-09-09-glyph-as-a-fold-execution.md` §S3). Each fold's
/// trip count — the JIT cache's key, and the unroll count `ExpandReduce`
/// reads — is now `pieces.next_power_of_two()`, not `pieces`, so `A`'s 11
/// pieces unroll as 16 and `O`'s 28 as 32; `8`'s piece count in this font
/// is already a power of two, so it pays nothing and its ceiling is
/// untouched. The padding rows are exact identities of both folds (pinned
/// by the padding-row test in `loop_blinn::tests`
/// and every coverage golden), so this is evaluated cost, not a coverage
/// change — measured, ~10% headroom, same ratchet as above.
///
/// Lowered 2026-09-23 to the measured count plus ~10% (142 → 156), when a
/// glyph became one fold whose body is an integral the e-graph closes
/// (`fonts/loop_blinn.rs`). The fold stays a loop to the assembler, so the
/// emitted program is the closed body once, whatever the piece count: all
/// three glyphs emit the same 142 nodes. The ceilings had not moved since
/// the legalizer stopped unrolling the fold before saturation — the glyphs
/// already emitted 155 nodes each against 2300, 4500 and 9100, a ratchet
/// nobody had turned.
///
/// One line-segment glyph, one all-quadratic, and one with many pieces
/// (6, 16 and 32 once horizontal pieces are dropped; `%` has the most in
/// ASCII, 40) — so a body that came to depend on the piece count again
/// would split them.
const CEILINGS: [(char, usize, usize); 3] = [('A', 16, 156), ('O', 16, 156), ('8', 32, 156)];

/// Three glyphs with a fivefold spread in piece count: 6, 16 and 32, three
/// trip-count buckets.
const SPREAD: [char; 3] = ['A', 'O', '8'];

/// The program the e-graph is handed must not grow with the glyph's piece
/// count. Ten pieces and forty are the same program over a different table.
///
/// Measured 100 nodes when the glyph became one fold of an integral, plus
/// ~10% (it was 165 with the winding and distance folds, against a ceiling
/// of 3200 set when the e-graph was handed the unrolled program).
const EGRAPH_INPUT_CEILING: usize = 110;

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
fn a_glyph_costs_no_more_than_it_did() {
    let font = Font::parse(FONT_BYTES).expect("parse font");
    let mut report = String::new();
    let mut over = Vec::new();

    for (ch, px, ceiling) in CEILINGS {
        let glyph = font
            .glyph_kernel_scaled(ch, px as f32)
            .unwrap_or_else(|| panic!("the font has no glyph for {ch:?}"));
        let coverage = glyph.kernel();
        let (arena, root) = coverage.parts();
        let shape = LatticeShape::new([px as u32, px as u32]);

        // The whole tier, as production runs it: link, legalize, saturate.
        // `None` is not a pass — it means the pipeline declined, and the
        // caller would then compile an arena that still holds a fold, which
        // the emitter has no instruction for.
        let optimized = pixelflow_search::runtime::optimize_runtime_arena(arena, root, shape)
            .unwrap_or_else(|| panic!("{ch}@{px}: the runtime pipeline declined"));
        let (out, out_root) = &*optimized;
        let nodes = reachable(out, *out_root);

        report.push_str(&format!("  {ch}@{px}: {nodes} nodes (ceiling {ceiling})\n"));
        if nodes > ceiling {
            over.push(format!("{ch}@{px}: {nodes} > {ceiling}"));
        }
    }

    assert!(
        over.is_empty(),
        "the runtime tier emits more than it used to for {}:\n{report}\n\
         Coverage being unchanged does not make this fine — the pixels are \
         pinned elsewhere, and this is the only test that reads the size of \
         the code that draws them. Note this is the LOOSE gate: if it failed \
         because the legalizer moved earlier in the pipeline, the fix is to \
         move it back, not to raise these.",
        over.join(", ")
    );
    println!("{report}");
}

/// **The gate that matters.** Saturation must be handed the glyph as
/// written — folds folded — so the program it reasons about is the same
/// size for `8` as for `A`, five times its piece count.
///
/// What this catches is any expansion creeping back in front of the e-graph:
/// unroll the fold first and this count becomes a multiple of the piece
/// count, which is how one glyph came to present 141,530 nodes to saturation.
///
/// It reads the arena rather than the pipeline, so it pins the *builder's*
/// half of the promise — a glyph presents a piece-count-independent program.
/// The pipeline's half (that nothing expands it before `Saturate`) is held by
/// the order written in `pixelflow_search::runtime`, with the emitted-size
/// test above as its tripwire.
#[test]
fn the_egraph_is_fed_a_program_it_can_reason_about() {
    let font = Font::parse(FONT_BYTES).expect("parse font");
    let mut sizes = Vec::new();

    for ch in SPREAD {
        let glyph = font
            .glyph_kernel_scaled(ch, 16.0)
            .unwrap_or_else(|| panic!("the font has no glyph for {ch:?}"));
        let kernel = glyph.kernel();
        // `Kernel::linked_parts` is the pipeline's one step before
        // `Saturate`: a `Ref` has no structure for the e-graph to read, so
        // it is resolved first. Everything after it is saturation's input.
        let (linked, lroot) = kernel.linked_parts();
        sizes.push((ch, reachable(&linked, lroot)));
    }

    let report: String = sizes
        .iter()
        .map(|(ch, n)| format!("  {ch}: {n} nodes\n"))
        .collect();

    for (ch, n) in &sizes {
        assert!(
            *n <= EGRAPH_INPUT_CEILING,
            "{ch}: saturation would be handed {n} nodes (ceiling \
             {EGRAPH_INPUT_CEILING}):\n{report}\
             An e-graph is superlinear in what it is fed. Something is \
             expanding the glyph before saturation sees it — legalization is \
             the LAST pass, and a fallback."
        );
    }

    let (min, max) = (
        sizes.iter().map(|(_, n)| *n).min().expect("three glyphs"),
        sizes.iter().map(|(_, n)| *n).max().expect("three glyphs"),
    );
    assert_eq!(
        min, max,
        "a glyph's program must not depend on its piece count — A, O and 8 \
         must present the same graph over different tables:\n{report}"
    );
    println!("{report}");
}
