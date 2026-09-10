//! Deployment and training of the **bilinear** saturation Guide — the arm
//! `docs/plans/2026-09-02-bilinear-guide-registration.md` §4 names as "the
//! thing under test", and the only implementation of [`SaturationGuide`]
//! whose rule-by-context second difference is not identically zero.
//!
//! # What this is, against what it is being compared to
//!
//! [`super::linear::LinearCandidateGuide`] scores
//! `s(r, x) = w_rule[r] + g(x)` — additively separable, so no context can
//! reorder two rules against each other (the claim, with its proof and its
//! five pinning tests, is in `scoring::representation`). This module deploys
//! `scoring::SaturationHead`'s tower instead:
//!
//! ```text
//! s(r, x) = m(x)ᵀ W r  +  bᵀ r
//! ```
//!
//! whose rule-pair difference `(m(x)ᵀW + bᵀ)(r₁ − r₂)` depends on `x`. Same
//! [`CandidateSummary`], same labels, same TRAIN split, same budget
//! denominator; the functional form is the licensed difference and the
//! registration disqualifies any other.
//!
//! # The three training decisions, stated rather than made silently
//!
//! **1. The op embeddings are FROZEN.** They carry the latency-prior
//! initialisation from #1063
//! ([`OpEmbeddings::new_with_latency_prior`]) and are a crate-shared asset,
//! not this head's private parameters. Two reasons beyond ownership, both
//! about the experiment rather than about taste:
//!
//! - *Redundancy.* The candidate tower's first layer is a full linear map
//!   `K → HIDDEN_DIM` applied to the pooled embedding, so training `E` and
//!   training `candidate_w1` move the same function through the same
//!   subspace. Adding `E` to the parameter set buys no expressiveness and
//!   costs a second copy of a shared object inside this checkpoint —
//!   "subtract before you add".
//! - *Confounding.* The additive arm has no op embedding at all; it has one
//!   scalar per op. Letting the bilinear arm re-learn a 50×32 embedding
//!   would be a capacity difference on the *context* side, and the
//!   registration licenses a difference only in the functional form.
//!
//! **2. The rule embedding IS trained — through its encoder, not as a free
//! table.** `SaturationHead::encode_rule`'s
//! `[LHS | RHS | LHS−RHS | LHS⊙RHS]` projection is what §4 registers, so
//! `rule_proj_w`/`rule_proj_b` carry the gradient and the per-rule vector
//! `r_j = P(c_j)` is derived. This is not a capacity handicap versus a free
//! per-rule table **provided** the 61 concatenations `c_j` are linearly
//! independent — then some `P` realizes any assignment of the `r_j`. That
//! is a checkable fact about this rule set, not an assumption:
//! [`BilinearTrainer::rule_encoding_rank`] computes it and the trainer
//! reports it. Deriving `r` also means an unseen rule still gets an
//! embedding rather than a zero, which a free table cannot do.
//!
//! **3. The side embeddings `z_LHS`/`z_RHS` are pooled op embeddings of the
//! rule's own templates** ([`ArenaRuleTemplate`]), `1/√n` scaled exactly as
//! the candidate tower pools a neighborhood. The expression tower that used
//! to produce them left with the extraction head (deleted 2026-09-01), and
//! pooling the template is the encoding that needs no new machinery. A rule
//! with no template on a side contributes a zero block there — reported by
//! the trainer, never silently absorbed.
//!
//! # Checkpoints carry parameters, never derived values
//!
//! [`BilinearWeights`] is the flat trained parameter vector, the frozen op
//! embeddings, and the rule-set [`Fingerprint`] — and nothing else. The
//! per-rule embeddings are **not** stored: they are `P(c_j)`, and both the
//! trainer and [`BilinearCandidateGuide::new`] derive them by the same code
//! from the same templates. Storing a derived value is storing a second
//! thing that can disagree with the first, and the mandatory skew test
//! (`skew_test_linear_guide --model bilinear`) is precisely a check that
//! this derivation agrees across the checkpoint boundary.
//!
//! As in [`super::linear`], parsing is *not* here: this module takes parsed
//! values and owns the **refusal** — a checkpoint whose fingerprint does not
//! match the live rule set is a load failure, never a warning and never a
//! default.

extern crate alloc;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use pixelflow_ir::expr::Term;

use super::linear::GuideError;
use super::scoring::backward::{Activations, HeadGradients};
use super::scoring::{RULE_CONCAT_DIM, SaturationHead, rule_concat};
use super::{CandidateSummary, SaturationGuide, shift_by};
use crate::egraph::rules::{Fingerprint, RuleId, RuleSet};
use crate::nnue::factored::{ArenaRuleTemplate, EMBED_DIM, K, OpEmbeddings};
use pixelflow_ir::OpKind;

/// A trained bilinear Guide's parameters, as a checkpoint carries them.
///
/// Three fields, deliberately: the trained parameters, the frozen inputs
/// they were trained against, and the identity of the vocabulary they name.
/// Everything else a scorer needs is derived — see the module doc.
#[derive(Clone, Debug)]
pub struct BilinearWeights {
    /// The trained parameters, flat, in the canonical order
    /// [`BilinearWeights::parameter_count`] counts. Produced by
    /// [`BilinearTrainer::weights`]; consumed only by
    /// [`BilinearCandidateGuide::new`].
    pub parameters: Vec<f32>,
    /// The frozen op embeddings, `OpKind::all().count() * K` floats,
    /// row-major in `OpKind::all()` order.
    ///
    /// Carried explicitly rather than reconstructed from a seed: they are an
    /// *input* to every score, and a later edit to the shared latency table
    /// would otherwise change a deployed checkpoint's behaviour without
    /// changing the checkpoint.
    pub op_embeddings: Vec<f32>,
    /// The [`RuleSet::fingerprint`] of the vocabulary these were trained
    /// against.
    pub fingerprint: Fingerprint,
}

impl BilinearWeights {
    /// How many floats [`Self::parameters`] must hold.
    #[must_use]
    pub fn parameter_count() -> usize {
        SaturationHead::trained_parameter_count()
    }

    /// How many floats [`Self::op_embeddings`] must hold.
    #[must_use]
    pub fn op_embedding_count() -> usize {
        OpKind::all().count() * K
    }
}

// ── Rule side encodings ─────────────────────────────────────────────────────

/// Pool a **depth-bound** op-embedding bag over every node reachable from
/// `root` in `arena`, `1/√n` scaled — the same `1/√n` the candidate tower
/// applies to a match's neighborhood, so a rule's side embedding and a
/// candidate's context embedding live at the same scale.
///
/// # Why depth-bound rather than a flat bag
///
/// A flat multiset of a template's ops is topology-blind, and this rule set
/// contains pairs that differ **only** in topology: `associative(Add)`
/// (`(A+B)+C → A+(B+C)`) and `reverse-associative(Add)` have the same op
/// multiset on each side, as do `exp-ln-cancel`/`ln-exp-cancel` and
/// `exp2-log2-cancel`/`log2-exp2-cancel`. A flat bag gives each such pair
/// **one** embedding, so the model cannot score them differently in any
/// context — a capacity gap the additive arm (one free scalar per rule) does
/// not have, which would confound the registered comparison.
///
/// Binding each op to its depth with [`shift_by`] — the generalised VSA
/// permutation `GraphAccumulator` already uses for exactly this purpose, so
/// this is a reuse and not a new mechanism — separates all three pairs. It
/// is not a change to the registered rule *encoding*: §4 registers the
/// 4-way `[LHS | RHS | LHS−RHS | LHS⊙RHS]` concatenation, and `z_LHS`/`z_RHS`
/// were left unspecified because the tower that used to produce them left
/// with the extraction head.
///
/// `None` (a side the rule does not define) pools to the zero vector, which
/// is the honest encoding of "this rule declares no such side" and is
/// visible to the caller as a zero block in the concatenation.
fn pool_template_side(embeddings: &OpEmbeddings, side: Option<Term<'_>>) -> [f32; EMBED_DIM] {
    let Some(side) = side else {
        return [0.0; EMBED_DIM];
    };
    let mut bound: Vec<[f32; K]> = Vec::new();
    let mut stack = alloc::vec![(side.root(), 0usize)];
    while let Some((node, depth)) = stack.pop() {
        bound.push(shift_by(
            embeddings.get(crate::nnue::factored::kind_of(node)),
            depth,
        ));
        stack.extend(node.children().map(|child| (child, depth + 1)));
    }
    let scale = 1.0 / libm::sqrtf(bound.len() as f32);
    let mut out = [0.0f32; EMBED_DIM];
    for e in bound {
        for i in 0..K {
            out[i] += e[i] * scale;
        }
    }
    out
}

/// Every rule's `[LHS | RHS | LHS−RHS | LHS⊙RHS]` concatenation, keyed by
/// stable identity, plus the labels of the rules that defined neither
/// template side.
///
/// A rule with neither side gets an all-zero concatenation, which projects
/// to the constant `rule_proj_b` — i.e. it becomes indistinguishable from
/// every other such rule. That is a real limitation of the registered
/// encoding, so it is returned rather than hidden, and the trainer prints
/// it.
fn rule_concats(
    embeddings: &OpEmbeddings,
    rules: &RuleSet,
) -> (
    BTreeMap<RuleId, [f32; RULE_CONCAT_DIM]>,
    Vec<alloc::string::String>,
) {
    let (shared, ids) = rules.shared();
    let mut map = BTreeMap::new();
    let mut templateless = Vec::new();
    for (idx, rule) in shared.iter().enumerate() {
        let template = ArenaRuleTemplate::from_rule(rule.as_ref());
        if template.lhs().is_none() && template.rhs().is_none() {
            if let Some(label) = rules.label_of(idx) {
                templateless.push(label);
            }
        }
        let z_lhs = pool_template_side(embeddings, template.lhs());
        let z_rhs = pool_template_side(embeddings, template.rhs());
        map.insert(ids[idx], rule_concat(&z_lhs, &z_rhs));
    }
    (map, templateless)
}

/// Rank of the matrix whose rows are the rule concatenations, by Gaussian
/// elimination with partial pivoting in `f64`.
///
/// Reported, not asserted: a rank below the rule count is a real statement
/// about this rule vocabulary (two rules whose templates pool identically
/// cannot be told apart by this encoder), and the honest response is to
/// print it next to the trained metrics, not to refuse to train.
fn concat_rank(concats: &BTreeMap<RuleId, [f32; RULE_CONCAT_DIM]>) -> usize {
    let mut rows: Vec<[f64; RULE_CONCAT_DIM]> = concats
        .values()
        .map(|c| {
            let mut r = [0.0f64; RULE_CONCAT_DIM];
            for i in 0..RULE_CONCAT_DIM {
                r[i] = f64::from(c[i]);
            }
            r
        })
        .collect();

    // Scale-relative pivot tolerance: the concatenations are O(1), so a
    // pivot below this is numerical dust, not a direction.
    const PIVOT_TOL: f64 = 1e-9;
    let mut rank = 0usize;
    for col in 0..RULE_CONCAT_DIM {
        let Some(pivot) = (rank..rows.len())
            .max_by(|&a, &b| {
                rows[a][col]
                    .abs()
                    .partial_cmp(&rows[b][col].abs())
                    .expect("finite concatenation entries")
            })
            .filter(|&p| rows[p][col].abs() > PIVOT_TOL)
        else {
            continue;
        };
        rows.swap(rank, pivot);
        let inv = 1.0 / rows[rank][col];
        for r in (rank + 1)..rows.len() {
            let factor = rows[r][col] * inv;
            if factor == 0.0 {
                continue;
            }
            for c in col..RULE_CONCAT_DIM {
                rows[r][c] -= factor * rows[rank][c];
            }
        }
        rank += 1;
        if rank == rows.len() {
            break;
        }
    }
    rank
}

// ── Deployment ──────────────────────────────────────────────────────────────

/// Deploys a trained [`BilinearWeights`] as a [`SaturationGuide`].
///
/// Scores are raw bilinear scores, no squashing:
/// [`SaturationGuide::score_candidates`] needs a move-ordering rank and the
/// guided loop sorts descending, so any monotone transform is wasted work.
#[derive(Clone)]
pub struct BilinearCandidateGuide {
    embeddings: OpEmbeddings,
    head: SaturationHead,
    rule_embeds: BTreeMap<RuleId, [f32; EMBED_DIM]>,
    fingerprint: Fingerprint,
}

/// Validate a checkpoint against the live rule set and unflatten it into the
/// two objects that score: the frozen op embeddings and the head.
///
/// One definition, shared by [`BilinearCandidateGuide::new`] and
/// [`BilinearTrainer::from_weights`] — a second copy of these three refusals
/// is a second place for one of them to be forgotten, and the skew test
/// compares exactly those two constructions.
fn load_parts(
    weights: &BilinearWeights,
    rules: &RuleSet,
) -> Result<(OpEmbeddings, SaturationHead), GuideError> {
    let live = rules.fingerprint();
    if weights.fingerprint != live {
        return Err(GuideError::new(alloc::format!(
            "bilinear guide: weights were trained against rule set {} but are being \
             deployed against {live} — the vocabulary changed since training, so every \
             derived rule embedding would name a different rule. Retrain rather than \
             deploy.",
            weights.fingerprint
        )));
    }
    let want_params = BilinearWeights::parameter_count();
    if weights.parameters.len() != want_params {
        return Err(GuideError::new(alloc::format!(
            "bilinear guide: checkpoint carries {} parameters, this build's head has \
             {want_params} — the checkpoint was written by a differently-shaped head; \
             retrain rather than deploy",
            weights.parameters.len()
        )));
    }
    let want_emb = BilinearWeights::op_embedding_count();
    if weights.op_embeddings.len() != want_emb {
        return Err(GuideError::new(alloc::format!(
            "bilinear guide: checkpoint carries {} op-embedding floats, this build's \
             OpKind table needs {want_emb} — the checkpoint was trained against a \
             different OpKind table; retrain or rebuild against a matching \
             pixelflow-ir revision",
            weights.op_embeddings.len()
        )));
    }

    let mut embeddings = OpEmbeddings::new();
    for (i, op) in OpKind::all().enumerate() {
        embeddings.e[op].copy_from_slice(&weights.op_embeddings[i * K..(i + 1) * K]);
    }
    let mut head = SaturationHead::new();
    head.load_trained_parameters(&weights.parameters);
    Ok((embeddings, head))
}

/// Identity, not contents: 16,000 weights printed into a panic message is
/// not a debugging aid, and the two things that actually distinguish two
/// deployed guides are the vocabulary they name and how many rules they
/// derived an embedding for.
impl core::fmt::Debug for BilinearCandidateGuide {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "BilinearCandidateGuide {{ fingerprint: {}, rules: {} }}",
            self.fingerprint,
            self.rule_embeds.len()
        )
    }
}

impl BilinearCandidateGuide {
    /// Deploy `weights` against `rules`, refusing them if they were trained
    /// against a different vocabulary or are the wrong shape.
    ///
    /// # Errors
    ///
    /// - the fingerprint disagrees with `rules.fingerprint()` — the
    ///   vocabulary changed since training, so every rule embedding this
    ///   would derive would name a rule the weights were never fit to;
    /// - `parameters` or `op_embeddings` is the wrong length — a truncated,
    ///   half-copied, or differently-shaped checkpoint, which would
    ///   otherwise deploy as a model with whatever the tail happened to be.
    pub fn new(weights: &BilinearWeights, rules: &RuleSet) -> Result<Self, GuideError> {
        let (embeddings, head) = load_parts(weights, rules)?;

        // Derived, not stored — see the module doc.
        let (concats, _) = rule_concats(&embeddings, rules);
        let rule_embeds = concats
            .iter()
            .map(|(&id, c)| (id, head.project_rule(c)))
            .collect();

        Ok(Self {
            embeddings,
            head,
            rule_embeds,
            fingerprint: rules.fingerprint(),
        })
    }

    /// The vocabulary these weights were trained against.
    #[must_use]
    pub fn fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }
}

impl SaturationGuide for BilinearCandidateGuide {
    fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32> {
        candidates
            .iter()
            .map(|c| self.head.score_candidate(&self.embeddings, c))
            .collect()
    }

    fn rule_embed(&self, rule: RuleId) -> [f32; EMBED_DIM] {
        *self.rule_embeds.get(&rule).unwrap_or_else(|| {
            panic!(
                "BilinearCandidateGuide: rule {rule} has no derived embedding — the \
                 candidate came from a rule set this guide was not built against, which \
                 `BilinearCandidateGuide::new`'s fingerprint check should have refused"
            )
        })
    }
}

// ── Training ────────────────────────────────────────────────────────────────

/// Positive offset applied to every ReLU bias at cold start — see
/// `SaturationHead::warm_relu_biases` for why a dead mask MLP is not a
/// generic training nuisance here but a silent collapse into the additive
/// model class this head is being compared against.
///
/// Sized against the pre-activation scale the head actually sees at init
/// (the mask MLP's hidden pre-activations are O(0.4) on cold-start
/// candidates), so it opens the layer without swamping the signal.
const RELU_WARM_BIAS: f32 = 0.5;

/// One SGD step's hyper-parameters.
///
/// A struct rather than two `f32` arguments: `step(c, d, 0.01, 1e-4)` has two
/// same-typed positional numbers whose swap is silent, and this head is
/// trained by a binary in another crate that reads them from two different
/// command-line flags.
#[derive(Clone, Copy, Debug)]
pub struct SgdStep {
    /// Learning rate.
    pub lr: f32,
    /// L2 weight decay, applied to weights (never biases) whether or not
    /// this step's sample touched them.
    pub l2: f32,
    /// Upper bound on the accumulated gradient's Euclidean norm; a larger
    /// gradient is rescaled to exactly this norm before the step.
    ///
    /// The additive arm clips `dLoss/dz` instead
    /// (`train_guide --grad-clip`), which is the right primitive for a
    /// one-layer convex model whose update is `lr · dz · x`. This head is
    /// four layers deep with a bilinear top, so the same `dz` produces
    /// wildly different weight-space step sizes depending on the
    /// activations, and bounding `dz` bounds nothing that matters. Clipping
    /// the *norm of the step* is the primitive that does. Both arms keep
    /// the identical loss and the identical class weighting; this bounds the
    /// optimiser, not the objective.
    pub max_grad_norm: f32,
}

/// One candidate's captured forward pass, held between scoring it and
/// accumulating its gradient.
///
/// Opaque, and it carries the candidate's [`RuleId`] rather than leaving the
/// caller to pass the same candidate to `accumulate` a second time: it
/// exists so a training loop computes the forward pass **once** (a loss
/// gradient needs the score, and the backward pass needs the activations
/// that produced *that* score, not a re-derived set), and pairing the wrong
/// candidate with a set of activations would train the rule projection of
/// one rule against the context of another.
pub struct Forward {
    activations: Activations,
    rule: RuleId,
}

impl Forward {
    /// The score this forward pass produced — identical to
    /// [`BilinearTrainer::score`] on the same candidate.
    #[must_use]
    pub fn score(&self) -> f32 {
        self.activations.score()
    }
}

/// The bilinear Guide's trainer: the one seam a supervised training loop in
/// another crate fills in.
///
/// Deliberately small. The head's weights, the mask MLP, the bilinear
/// scorer and the rule projection all stay private machinery behind
/// [`SaturationGuide`] (see [`super`]'s module doc); what is public is
/// forward, accumulate, apply, and a way to get a checkpoint out.
pub struct BilinearTrainer {
    embeddings: OpEmbeddings,
    head: SaturationHead,
    concats: BTreeMap<RuleId, [f32; RULE_CONCAT_DIM]>,
    templateless: Vec<alloc::string::String>,
    grads: Box<HeadGradients>,
    fingerprint: Fingerprint,
}

impl BilinearTrainer {
    /// A cold-started trainer over `rules`: op embeddings at the shared
    /// latency-prior initialisation (frozen thereafter) and head weights at
    /// `SaturationHead::randomize(seed)`.
    ///
    /// Cold start, per the registration's training protocol (§9) and the
    /// design doc's own bullet: no warm start from any prior guide or mask
    /// checkpoint. Head weights are *randomized* rather than zeroed because
    /// an all-zero bilinear is a saddle — `∂score/∂W = m ⊗ r` vanishes when
    /// `r` is zero and `∂score/∂r` vanishes when `W` and `b` are — so a
    /// zero init would train nothing for many steps and then train from
    /// wherever numerical noise pushed it. The additive model's all-zero
    /// cold start has no such saddle; this is the same *protocol* (no warm
    /// start) adapted to a model that has a symmetry the linear one lacks,
    /// and it is stated here rather than left as a magic constant.
    #[must_use]
    pub fn new_cold(rules: &RuleSet, seed: u64) -> Self {
        let embeddings = OpEmbeddings::new_with_latency_prior(seed);
        let mut head = SaturationHead::new();
        head.randomize(seed);
        head.warm_relu_biases(RELU_WARM_BIAS);
        let (concats, templateless) = rule_concats(&embeddings, rules);
        Self {
            embeddings,
            head,
            concats,
            templateless,
            grads: HeadGradients::zero(),
            fingerprint: rules.fingerprint(),
        }
    }

    /// Rebuild a trainer from a checkpoint, for a consumer that needs the
    /// **trainer's** forward pass over already-trained weights — the
    /// mandatory skew test, which has to ask the trainer and the deployed
    /// guide the same question about the same file.
    ///
    /// Not a warm start: nothing in this program resumes training from a
    /// checkpoint (the registration's protocol is cold start, §9), and this
    /// exists so the skew test does not have to re-run training to obtain a
    /// trainer that holds the trained head.
    ///
    /// # Errors
    ///
    /// The same three refusals [`BilinearCandidateGuide::new`] makes, for
    /// the same reasons.
    pub fn from_weights(weights: &BilinearWeights, rules: &RuleSet) -> Result<Self, GuideError> {
        let (embeddings, head) = load_parts(weights, rules)?;
        let (concats, templateless) = rule_concats(&embeddings, rules);
        Ok(Self {
            embeddings,
            head,
            concats,
            templateless,
            grads: HeadGradients::zero(),
            fingerprint: rules.fingerprint(),
        })
    }

    /// The labels of rules that define neither template side, and so encode
    /// to the same all-zero concatenation as each other.
    #[must_use]
    pub fn templateless_rules(&self) -> &[alloc::string::String] {
        &self.templateless
    }

    /// Pairs of rules this encoder cannot tell apart — their concatenations
    /// are bit-identical, so they score identically in every context, at any
    /// weights.
    ///
    /// A real limitation of the registered rule encoding, returned so a
    /// trainer reports it next to its metrics rather than absorbing it. On
    /// the production vocabulary the only such pair is the two rules that
    /// define no template at all; every rule with a template is separated,
    /// which is what `pool_template_side`'s depth binding buys.
    #[must_use]
    pub fn indistinguishable_rules(
        &self,
        rules: &RuleSet,
    ) -> Vec<(alloc::string::String, alloc::string::String)> {
        let label = |id: RuleId| {
            rules
                .index_of(id)
                .and_then(|i| rules.label_of(i))
                .unwrap_or_else(|| alloc::format!("<rule {id}>"))
        };
        let ids: Vec<RuleId> = self.concats.keys().copied().collect();
        let mut out = Vec::new();
        for (i, &a) in ids.iter().enumerate() {
            for &b in &ids[i + 1..] {
                if self.concats[&a] == self.concats[&b] {
                    out.push((label(a), label(b)));
                }
            }
        }
        out
    }

    /// Rank of the rule-concatenation matrix. Equal to the rule count means
    /// the encoder can express any per-rule embedding assignment — i.e. this
    /// arm is not handicapped against a free per-rule table (module doc,
    /// decision 2).
    #[must_use]
    pub fn rule_encoding_rank(&self) -> usize {
        concat_rank(&self.concats)
    }

    /// How many rules this trainer has an encoding for.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.concats.len()
    }

    /// This rule's current embedding, `P(c_rule)` — the value a caller must
    /// put in [`CandidateSummary::rule_embed`] before scoring, and exactly
    /// what [`BilinearCandidateGuide::rule_embed`] will answer once the
    /// checkpoint is deployed.
    ///
    /// # Panics
    ///
    /// If `rule` is not in the vocabulary this trainer was built over — a
    /// sample naming a rule the model has no encoding for is a dataset /
    /// rule-set mismatch, and scoring it against a fabricated zero embedding
    /// would train a real weight against a made-up input.
    #[must_use]
    pub fn rule_embed(&self, rule: RuleId) -> [f32; EMBED_DIM] {
        let concat = self.concats.get(&rule).unwrap_or_else(|| {
            panic!(
                "BilinearTrainer: rule {rule} is not in the vocabulary this trainer was \
                 built over ({} rules) — the dataset was minted against a different rule \
                 set",
                self.concats.len()
            )
        });
        self.head.project_rule(concat)
    }

    /// Run the forward pass, keeping the activations for
    /// [`Self::accumulate`].
    ///
    /// `candidate.rule_embed` must be [`Self::rule_embed`] of
    /// `candidate.rule`; it is checked, not assumed, because that equality
    /// is the whole train/deploy contract of this head.
    #[must_use]
    pub fn forward(&self, candidate: &CandidateSummary) -> Forward {
        let want = self.rule_embed(candidate.rule);
        assert!(
            want.iter().all(|v| v.is_finite()),
            "BilinearTrainer: rule {}'s embedding is not finite — the head has diverged; \
             this is reported here rather than as a mismatch below, because a NaN is never \
             equal to itself and would otherwise surface as a confusing skew error",
            candidate.rule
        );
        assert!(
            want == candidate.rule_embed,
            "BilinearTrainer: candidate for rule {} carries a rule_embed this trainer did \
             not produce — a training loop must set `rule_embed` from \
             `BilinearTrainer::rule_embed` (the same value the deployed guide answers), or \
             the model is trained against one encoding and deployed with another",
            candidate.rule
        );
        Forward {
            activations: Activations::forward(&self.head, &self.embeddings, candidate),
            rule: candidate.rule,
        }
    }

    /// The deployed score of `candidate` — the same computation
    /// [`BilinearCandidateGuide::score_candidates`] performs, for the skew
    /// test and for held-out evaluation.
    #[must_use]
    pub fn score(&self, candidate: &CandidateSummary) -> f32 {
        self.head.score_candidate(&self.embeddings, candidate)
    }

    /// Accumulate `d_score * dscore/dw` for every trained weight, including
    /// the rule projection, where `d_score` is `dLoss/dscore`.
    ///
    /// Adds to whatever is already accumulated; [`Self::apply`] consumes and
    /// clears. This trainer knows nothing about the loss — which loss, and
    /// which class weighting, is the calling binary's decision and belongs
    /// in its report.
    pub fn accumulate(&mut self, forward: &Forward, d_score: f32) {
        let d_rule = self
            .head
            .backward(&forward.activations, d_score, &mut self.grads);
        let concat = self.concats.get(&forward.rule).unwrap_or_else(|| {
            panic!(
                "BilinearTrainer: rule {} has no encoding — `forward` should have refused \
                 this candidate first",
                forward.rule
            )
        });
        self.grads.accumulate_rule_proj(concat, &d_rule);
    }

    /// Apply one SGD step from the accumulated gradients, then clear them.
    ///
    /// Returns the gradient's pre-clip Euclidean norm, so a training loop
    /// can report the clip rate rather than clip silently.
    ///
    /// # Panics
    ///
    /// If the accumulated gradient is not finite. A NaN reaching the weights
    /// makes every later score NaN, every later gradient NaN, and every
    /// later metric a quiet zero — the run would finish and write a
    /// checkpoint. This is the one place that is cheap to check, because the
    /// norm has to be computed for clipping anyway.
    pub fn apply(&mut self, step: SgdStep) -> f32 {
        let norm = self.grads.l2_norm();
        assert!(
            norm.is_finite(),
            "BilinearTrainer::apply: the accumulated gradient is not finite ({norm}) — \
             training has diverged; lower the learning rate or the gradient-norm bound \
             rather than writing this checkpoint"
        );
        if norm > step.max_grad_norm && norm > 0.0 {
            self.grads.scale(step.max_grad_norm / norm);
        }
        self.head.apply_sgd(&self.grads, step.lr, step.l2);
        self.grads.clear();
        norm
    }

    /// The trained parameters as a checkpoint carries them.
    #[must_use]
    pub fn weights(&self) -> BilinearWeights {
        let mut op_embeddings = Vec::with_capacity(BilinearWeights::op_embedding_count());
        for op in OpKind::all() {
            op_embeddings.extend_from_slice(self.embeddings.get(op));
        }
        BilinearWeights {
            parameters: self.head.trained_parameters(),
            op_embeddings,
            fingerprint: self.fingerprint,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn candidate(trainer: &BilinearTrainer, rule: RuleId, ops: Vec<OpKind>) -> CandidateSummary {
        CandidateSummary {
            rule_embed: trainer.rule_embed(rule),
            neighborhood_ops: ops,
            budget_fraction: 0.3,
            rule,
            match_class_node_count: 4,
            expr_node_count: 17,
        }
    }

    fn first_two_rules(rules: &RuleSet) -> (RuleId, RuleId) {
        (
            rules.id_of(0).expect("a first rule"),
            rules.id_of(1).expect("a second rule"),
        )
    }

    #[test]
    fn a_trained_checkpoint_should_round_trip_bit_exactly_through_the_deployed_guide() {
        // The property the mandatory skew test measures on real DEV records,
        // stated here as a unit test so a layout bug fails in `cargo test`
        // rather than in a training run's last step.
        let rules = RuleSet::production();
        let mut trainer = BilinearTrainer::new_cold(&rules, 11);
        let (r0, r1) = first_two_rules(&rules);

        // Take a few real steps so the checkpoint is not the init.
        for i in 0..8 {
            let rule = if i % 2 == 0 { r0 } else { r1 };
            let c = candidate(&trainer, rule, vec![OpKind::Sin, OpKind::Add]);
            let f = trainer.forward(&c);
            trainer.accumulate(&f, 0.5);
            trainer.apply(SgdStep {
                lr: 0.01,
                l2: 1e-4,
                max_grad_norm: 1.0,
            });
        }

        let weights = trainer.weights();
        let guide = BilinearCandidateGuide::new(&weights, &rules).expect("same vocabulary");

        for rule in [r0, r1] {
            assert_eq!(
                guide.rule_embed(rule),
                trainer.rule_embed(rule),
                "the deployed guide must derive the same rule embedding the trainer used"
            );
            let c = candidate(&trainer, rule, vec![OpKind::Sqrt, OpKind::Mul, OpKind::Sin]);
            let deployed = guide.score_candidates(core::slice::from_ref(&c))[0];
            assert_eq!(
                deployed,
                trainer.score(&c),
                "trainer and deployed score must agree bit for bit"
            );
        }
    }

    #[test]
    fn new_should_refuse_weights_from_a_different_rule_vocabulary() {
        let rules = RuleSet::production();
        let trainer = BilinearTrainer::new_cold(&rules, 3);
        let mut weights = trainer.weights();

        let other = RuleSet::new(vec![crate::math::algebra::Commutative::new(
            &crate::egraph::ops::Add,
        )]);
        assert_ne!(other.fingerprint(), rules.fingerprint());
        weights.fingerprint = other.fingerprint();

        let err = BilinearCandidateGuide::new(&weights, &rules)
            .expect_err("a fingerprint mismatch must be refused, never defaulted");
        assert!(
            alloc::format!("{err}").contains("vocabulary changed since training"),
            "the refusal must say what went wrong: {err}"
        );
    }

    #[test]
    fn new_should_refuse_a_wrong_length_parameter_vector() {
        let rules = RuleSet::production();
        let trainer = BilinearTrainer::new_cold(&rules, 3);
        let mut weights = trainer.weights();
        weights.parameters.pop();
        let err = BilinearCandidateGuide::new(&weights, &rules)
            .expect_err("a truncated checkpoint must be refused");
        assert!(alloc::format!("{err}").contains("parameters"), "{err}");

        let mut weights = trainer.weights();
        weights.op_embeddings.push(0.0);
        let err = BilinearCandidateGuide::new(&weights, &rules)
            .expect_err("a wrong-shaped op-embedding block must be refused");
        assert!(alloc::format!("{err}").contains("op-embedding"), "{err}");
    }

    #[test]
    fn forward_should_refuse_a_candidate_whose_rule_embed_it_did_not_produce() {
        let rules = RuleSet::production();
        let trainer = BilinearTrainer::new_cold(&rules, 5);
        let (r0, _) = first_two_rules(&rules);
        let mut c = candidate(&trainer, r0, vec![OpKind::Add]);
        c.rule_embed[0] += 1.0;
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = trainer.forward(&c);
        }));
        assert!(
            panicked.is_err(),
            "a rule_embed the trainer did not produce is train/deploy skew and must be loud"
        );
    }

    #[test]
    fn the_rule_encoder_should_separate_every_rule_that_has_a_template() {
        // Module doc, decision 2. The bilinear arm must not be handicapped
        // against the additive arm's free per-rule scalar, and the sharp
        // version of that is: no two rules may share an encoding, because a
        // shared encoding means one score in every context at any weights.
        //
        // Exactly one exception exists in the production vocabulary and it
        // is structural rather than accidental — two rules define **no**
        // template on either side, so the registered
        // `[LHS | RHS | LHS−RHS | LHS⊙RHS]` encoding has literally nothing
        // to encode and gives them both the zero vector. That is recorded
        // as a limitation (the trainer prints it), not papered over, and
        // this test pins that it stays the *only* exception: a new rule that
        // collides with an existing one fails here rather than quietly
        // sharing its score.
        let rules = RuleSet::production();
        let trainer = BilinearTrainer::new_cold(&rules, 1);
        let templateless = trainer.templateless_rules();
        for (a, b) in trainer.indistinguishable_rules(&rules) {
            assert!(
                templateless.contains(&a) && templateless.contains(&b),
                "rules {a:?} and {b:?} encode identically but at least one of them has a \
                 template — the encoder has lost a distinction it can make (templateless: \
                 {templateless:?})"
            );
        }
    }

    #[test]
    fn the_rule_encoding_rank_should_be_reported_and_near_full() {
        // The weaker, quantitative half of the statement above: the rank of
        // the concatenation matrix is what decides whether *some* projection
        // realizes an arbitrary per-rule embedding assignment. Each
        // indistinguishable pair costs exactly one dimension; this pins that
        // the shortfall is no larger than that, so nothing else in the
        // encoding is silently degenerate.
        let rules = RuleSet::production();
        let trainer = BilinearTrainer::new_cold(&rules, 1);
        let collisions = trainer.indistinguishable_rules(&rules).len();
        let rank = trainer.rule_encoding_rank();
        assert!(
            rank + collisions + 1 >= trainer.rule_count(),
            "rule-encoding rank {rank} of {} with {collisions} identical pair(s) — more \
             rank is missing than the collisions explain, so the encoder is degenerate in \
             a way this module has not accounted for",
            trainer.rule_count()
        );
    }

    #[test]
    fn training_should_separate_a_rule_the_loss_pushes_up_from_one_it_pushes_down() {
        // The end-to-end training statement: with the same context, a rule
        // driven by a negative loss gradient must end up scoring above one
        // driven by a positive gradient. This is what makes the trainer a
        // trainer rather than a set of gradient formulas that happen to
        // agree with finite differences.
        let rules = RuleSet::production();
        let mut trainer = BilinearTrainer::new_cold(&rules, 17);
        let (up, down) = first_two_rules(&rules);
        let ops = vec![OpKind::Sin, OpKind::Cos];

        for _ in 0..300 {
            for (rule, d) in [(up, -1.0f32), (down, 1.0f32)] {
                let c = candidate(&trainer, rule, ops.clone());
                let f = trainer.forward(&c);
                trainer.accumulate(&f, d);
                trainer.apply(SgdStep {
                    lr: 0.01,
                    l2: 0.0,
                    max_grad_norm: 1.0,
                });
            }
        }

        let s_up = trainer.score(&candidate(&trainer, up, ops.clone()));
        let s_down = trainer.score(&candidate(&trainer, down, ops));
        assert!(
            s_up > s_down,
            "the rule trained upward must outrank the one trained downward: \
             {s_up} vs {s_down}"
        );
    }

    #[test]
    fn a_trained_bilinear_should_be_able_to_reorder_two_rules_between_two_contexts() {
        // The registration's whole point, as a *training* statement rather
        // than the hand-set weights `scoring::representation` pins: fit two
        // rules to opposite preferences in two contexts and check the
        // learned model reorders them. An additively separable score cannot
        // reach this configuration at any weights.
        let rules = RuleSet::production();
        let mut trainer = BilinearTrainer::new_cold(&rules, 23);
        let (a, b) = first_two_rules(&rules);
        let trig = vec![OpKind::Sin, OpKind::Cos];
        let poly = vec![OpKind::Add, OpKind::Mul];

        for _ in 0..800 {
            for (rule, ops, d) in [
                (a, &trig, -1.0f32),
                (b, &trig, 1.0),
                (a, &poly, 1.0),
                (b, &poly, -1.0),
            ] {
                let c = candidate(&trainer, rule, ops.clone());
                let f = trainer.forward(&c);
                trainer.accumulate(&f, d);
                trainer.apply(SgdStep {
                    lr: 0.02,
                    l2: 0.0,
                    max_grad_norm: 1.0,
                });
            }
        }

        let trig_gap = trainer.score(&candidate(&trainer, a, trig.clone()))
            - trainer.score(&candidate(&trainer, b, trig));
        let poly_gap = trainer.score(&candidate(&trainer, a, poly.clone()))
            - trainer.score(&candidate(&trainer, b, poly));
        assert!(
            trig_gap > 0.0 && poly_gap < 0.0,
            "training must be able to put rule A above B in one context and below it in \
             the other: trig gap {trig_gap}, poly gap {poly_gap}"
        );
    }
}
