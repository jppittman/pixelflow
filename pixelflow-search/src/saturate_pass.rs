//! Equality saturation as an [`Optimize`] value.
//!
//! [`Optimizer`] decides *how* saturation runs — rule set, budget, cost model,
//! extractor. [`Saturate`] is the endomorphism that runs it: insert the term,
//! saturate, extract, materialise. Those are different things, and until
//! [`Optimize`] existed only the first had a name, so every tier wrote the
//! second out by hand.

use pixelflow_ir::LatticeShape;
use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_ir::optimize::{Optimize, Rewritten};

use crate::egraph::{Optimizer, RuleSet, Vocabulary, insert, reachable_count};
use crate::tier::Tier;

/// Rewrite a term by equality saturation under `optimizer`.
///
/// Declines a term the [`Vocabulary`] cannot hold — a `Param` slot that
/// should have been specialized, an op this tier may not model. Declining is
/// ordinary: the caller compiles the term unoptimized.
pub struct Saturate {
    optimizer: Optimizer,
    vocab: Vocabulary,
    tier: Tier,
}

impl Saturate {
    /// The runtime tier's configuration: production policy priced against the
    /// lattice this kernel is compiled for — the extents are known, so
    /// extraction minimizes the instruction count of the whole program rather
    /// than of its text — over the runtime vocabulary.
    #[must_use]
    pub fn runtime(shape: LatticeShape) -> Self {
        Self {
            optimizer: Optimizer::production()
                .rules(RuleSet::runtime())
                .for_lattice(shape),
            vocab: Vocabulary::Runtime,
            tier: Tier::Runtime,
        }
    }

    /// The macro tier's configuration: the same production policy, over the
    /// template vocabulary, priced without a lattice — `kernel!` expands
    /// before any consumer has said what shape it wants.
    ///
    /// Sits here beside [`Self::runtime`] rather than in the compiler crate
    /// because the two are one decision: what differs between the tiers is
    /// this, and it should be readable in one place. It also carries
    /// [`Tier::Macro`], which is not bookkeeping — a macro-tier saturation
    /// writes to rustc's stderr, and telemetry has to prefix its record so
    /// cargo's `--message-format=json` parser does not read it as a compiler
    /// message.
    #[must_use]
    pub fn macro_tier() -> Self {
        Self {
            optimizer: Optimizer::production(),
            vocab: Vocabulary::Templates,
            tier: Tier::Macro,
        }
    }

    /// Saturation under an explicitly chosen policy and vocabulary, for
    /// harnesses that vary one and hold the rest.
    #[must_use]
    pub fn with(optimizer: Optimizer, vocab: Vocabulary, tier: Tier) -> Self {
        Self {
            optimizer,
            vocab,
            tier,
        }
    }
}

impl Optimize for Saturate {
    fn optimize(&mut self, arena: &ExprArena, root: ExprId) -> Rewritten {
        let mut egraph = self.optimizer.egraph();
        let Ok(root_class) = insert(arena, root, &mut egraph, self.vocab) else {
            return Rewritten::Declined;
        };

        let node_count = reachable_count(arena, root);
        #[cfg(feature = "saturation-telemetry")]
        let inserted_classes = egraph.num_classes();
        #[cfg(feature = "saturation-telemetry")]
        let telemetry_start = std::time::Instant::now();
        let optimized = self.optimizer.run(&mut egraph, root_class, node_count);
        let (extracted, extracted_root) = optimized.to_arena(&egraph, root_class);

        #[cfg(feature = "saturation-telemetry")]
        crate::telemetry::record(crate::telemetry::SaturationInvocation {
            tier: self.tier,
            node_count,
            inserted_classes,
            extraction: optimized.extraction,
            stats: &optimized.stats,
            union_count: optimized.stats.unions,
            extracted_arena: &extracted,
            extracted_root,
            wall_clock: telemetry_start.elapsed(),
            kernel_label: None,
        });

        // The extracted arena declares buffers in extraction-traversal order,
        // which need not match the input's — and slot order is ABI: the JIT
        // loads slot i's base pointer from the caller's context array at i*8,
        // and callers bind in the order the arena THEY BUILT declared. A
        // different extraction (a commuted equivalent under another cost
        // model) must not silently permute their pointers. Re-splicing onto a
        // table pre-declared in input order makes the invariant structural:
        // splice dedups buffers by identity onto the existing slots.
        if arena.buffers().is_empty() {
            return Rewritten::Changed(extracted, extracted_root);
        }
        let mut ordered = ExprArena::new();
        for decl in arena.buffers() {
            let _slot = ordered.declare_buffer(*decl);
        }
        let new_root = ordered.splice(&extracted, extracted_root);
        debug_assert!(
            ordered
                .buffers()
                .iter()
                .zip(arena.buffers())
                .all(|(a, b)| a.id == b.id),
            "buffer slot order must survive optimization"
        );
        Rewritten::Changed(ordered, new_root)
    }
}
