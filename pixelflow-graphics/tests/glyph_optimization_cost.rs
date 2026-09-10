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
use pixelflow_ir::{Environment, LatticeShape};

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
/// One line-segment glyph, one all-quadratic, and the one whose waist
/// tangency is the knife edge the class-cap sweep is blocked on
/// (`egraph::saturate::CLASSICAL_CLASS_CEILING`) — so if that unblocks and
/// the cap rises, this notices.
const CEILINGS: [(char, usize, usize); 3] = [('A', 16, 1600), ('O', 16, 4000), ('8', 32, 9100)];

/// `(character, piece count)` — 11, 28 and 34 pieces, a 3× spread.
const SPREAD: [char; 3] = ['A', 'O', '8'];

/// The program the e-graph is handed must not grow with the glyph's piece
/// count. Ten pieces and forty are the same program over a different table.
const EGRAPH_INPUT_CEILING: usize = 3200;

fn reachable(root: pixelflow_ir::Node<'_, pixelflow_ir::ExprData>) -> usize {
    root.node_count()
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
        let env = Environment {
            buffers: coverage.buffers().to_vec(),
            uniforms: coverage.uniforms().to_vec(),
        };
        let shape = LatticeShape::new([px as u32, px as u32]);

        // The whole tier, as production runs it: link, legalize, saturate.
        // `None` is not a pass — it means the pipeline declined, and the
        // caller would then compile an arena that still holds a fold, which
        // the emitter has no instruction for.
        let optimized = pixelflow_search::runtime::optimize_runtime_dag(
            coverage.rooted(),
            &env,
            shape,
        )
        .unwrap_or_else(|| panic!("{ch}@{px}: the runtime pipeline declined"));
        let nodes = reachable(optimized.0.entry());

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
/// size for a 34-piece glyph as for an 11-piece one.
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
        let env = Environment {
            buffers: kernel.buffers().to_vec(),
            uniforms: kernel.uniforms().to_vec(),
        };
        // `ExpandRefs` is the pipeline's one step before `Saturate`: a `Ref`
        // has no structure for the e-graph to read, so it is resolved first.
        // Everything after it is saturation's input.
        let (linked, _) = pixelflow_ir::passes::expand_refs_rooted(kernel.rooted(), &env);
        sizes.push((ch, reachable(linked.entry())));
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
        "a glyph's program must not depend on its piece count — 11, 28 and 34 \
         pieces must present the same graph over a different table:\n{report}"
    );
    println!("{report}");
}
