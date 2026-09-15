//! The candidate-scoring path's backward pass: gradients of
//! [`SaturationHead::score_candidate`] with respect to every weight on that
//! path, and the SGD step that applies them.
//!
//! # What is differentiated, and what is not
//!
//! Exactly the tensors [`SaturationHead::score_candidate`] reads:
//!
//! ```text
//! input  = candidate_input(embeddings, candidate)          (K pooled op lanes + 4 scalars)
//! h1     = relu(candidate_b1     + inputᵀ candidate_w1)
//! h2     = relu(trunk_b          + h1ᵀ    trunk_w)
//! e      =      candidate_proj_b + h2ᵀ    candidate_proj_w
//! g      = relu(mask_mlp_b1      + eᵀ     mask_mlp_w1)
//! m      =      mask_mlp_b2      + gᵀ     mask_mlp_w2
//! r      =      rule_proj_b      + concatᵀ rule_proj_w
//! score  = (mᵀ interaction + mask_bias_projᵀ) · r
//! ```
//!
//! **Trained:** `candidate_w1/b1`, `trunk_w/b`, `candidate_proj_w/b`,
//! `mask_mlp_w1/b1`, `mask_mlp_w2/b2`, `interaction`, `mask_bias_proj`, and
//! `rule_proj_w/b` — the rule embedding *through its encoder*, which is how
//! the registration (§4) names it, rather than as a free per-rule table.
//!
//! **Frozen, deliberately:** the [`OpEmbeddings`] the neighborhood is pooled
//! in, and the `graph_*` tower (which the candidate path never reads at all,
//! so it has no gradient here to freeze — it is simply absent from
//! [`HeadGradients`], which is why that struct is not shaped like the head).
//! The op-embedding decision is stated in [`super::super::bilinear`]'s module
//! doc, where the trainer that acts on it lives.
//!
//! # Why the gradients are checked numerically, not read
//!
//! A bilinear layer is where a transposed index survives review: `interaction`
//! is square (`EMBED_DIM × EMBED_DIM`), so `m[i]·dt[j]` and `m[j]·dt[i]` are
//! both well-typed, both finite, and both plausible in a loss curve — the
//! wrong one just trains a transposed model, silently. Every tensor below is
//! therefore pinned by a central-difference check against the forward pass
//! itself (`numerical_gradient_check_*`, in the pattern
//! `nnue::factored`'s `numerical_gradient_check_embeddings` established), on
//! a fixture whose weights are *asymmetric* so a transpose cannot pass.

extern crate alloc;

use super::{
    CANDIDATE_INPUT_DIM, EMBED_DIM, HIDDEN_DIM, MLP_HIDDEN, OpEmbeddings, RULE_CONCAT_DIM,
    SaturationHead, candidate_input,
};
use crate::nnue::guide::CandidateSummary;

/// Every activation [`SaturationHead::backward`] needs, captured on the way
/// forward.
///
/// Post-ReLU values are stored rather than pre-activations: for a ReLU,
/// `out > 0` iff the pre-activation was positive, so the post-activation
/// carries the mask as well as the value, and storing one array instead of
/// two removes the possibility of the two disagreeing.
pub(crate) struct Activations {
    input: [f32; CANDIDATE_INPUT_DIM],
    /// Candidate tower hidden, post-ReLU.
    h1: [f32; HIDDEN_DIM],
    /// Trunk hidden, post-ReLU.
    h2: [f32; HIDDEN_DIM],
    /// Candidate embedding (no activation).
    embed: [f32; EMBED_DIM],
    /// Mask MLP hidden, post-ReLU.
    g: [f32; MLP_HIDDEN],
    /// Mask features (no activation).
    mask_features: [f32; EMBED_DIM],
    /// `maskᵀ interaction` — the bilinear's context-side vector.
    transformed: [f32; EMBED_DIM],
    /// The rule embedding this candidate was scored against
    /// ([`CandidateSummary::rule_embed`]).
    rule_embed: [f32; EMBED_DIM],
    /// The score itself, so a caller never recomputes it (and so cannot
    /// compute a loss gradient at a different point than the one the
    /// activations describe).
    score: f32,
}

impl Activations {
    /// Run [`SaturationHead::score_candidate`]'s forward pass, keeping every
    /// intermediate.
    ///
    /// The score this returns is the score `score_candidate` returns for the
    /// same inputs — pinned by
    /// `activations_score_should_equal_score_candidate`.
    #[must_use]
    pub(crate) fn forward(
        head: &SaturationHead,
        embeddings: &OpEmbeddings,
        candidate: &CandidateSummary,
    ) -> Self {
        let input = candidate_input(embeddings, candidate);

        let mut h1 = head.candidate_b1;
        for (i, &v) in input.iter().enumerate() {
            for (j, h) in h1.iter_mut().enumerate() {
                *h += v * head.candidate_w1[i][j];
            }
        }
        for h in &mut h1 {
            *h = h.max(0.0);
        }

        let h2 = head.apply_trunk(&h1);
        let embed = head.compute_candidate_embed(&h2);

        let mut g = head.mask_mlp_b1;
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                g[j] += embed[i] * head.mask_mlp_w1[i][j];
            }
        }
        for v in &mut g {
            *v = v.max(0.0);
        }

        let mut mask_features = head.mask_mlp_b2;
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                mask_features[k] += g[j] * head.mask_mlp_w2[j][k];
            }
        }

        let mut transformed = [0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                transformed[j] += mask_features[i] * head.interaction[i][j];
            }
        }

        let rule_embed = candidate.rule_embed;
        let mut score = 0.0f32;
        for k in 0..EMBED_DIM {
            score += (transformed[k] + head.mask_bias_proj[k]) * rule_embed[k];
        }

        Self {
            input,
            h1,
            h2,
            embed,
            g,
            mask_features,
            transformed,
            rule_embed,
            score,
        }
    }

    /// The forward score these activations were captured at.
    #[must_use]
    pub(crate) fn score(&self) -> f32 {
        self.score
    }
}

/// Accumulated gradients for the tensors on the candidate-scoring path.
///
/// Not shaped like [`SaturationHead`]: the `graph_*` tower is absent because
/// `score_candidate` never reads it, and a zero gradient array for a tensor
/// nothing differentiates is 8,448 floats of invitation to "train" it by
/// accident.
pub(crate) struct HeadGradients {
    pub(crate) candidate_w1: [[f32; HIDDEN_DIM]; CANDIDATE_INPUT_DIM],
    pub(crate) candidate_b1: [f32; HIDDEN_DIM],
    pub(crate) trunk_w: [[f32; HIDDEN_DIM]; HIDDEN_DIM],
    pub(crate) trunk_b: [f32; HIDDEN_DIM],
    pub(crate) candidate_proj_w: [[f32; EMBED_DIM]; HIDDEN_DIM],
    pub(crate) candidate_proj_b: [f32; EMBED_DIM],
    pub(crate) mask_mlp_w1: [[f32; MLP_HIDDEN]; EMBED_DIM],
    pub(crate) mask_mlp_b1: [f32; MLP_HIDDEN],
    pub(crate) mask_mlp_w2: [[f32; EMBED_DIM]; MLP_HIDDEN],
    pub(crate) mask_mlp_b2: [f32; EMBED_DIM],
    pub(crate) interaction: [[f32; EMBED_DIM]; EMBED_DIM],
    pub(crate) mask_bias_proj: [f32; EMBED_DIM],
    pub(crate) rule_proj_w: [[f32; EMBED_DIM]; RULE_CONCAT_DIM],
    pub(crate) rule_proj_b: [f32; EMBED_DIM],
}

impl HeadGradients {
    /// All-zero gradients — a fresh accumulator.
    #[must_use]
    pub(crate) fn zero() -> alloc::boxed::Box<Self> {
        // Boxed: ~59 KB of f32, and a by-value return would build it on the
        // caller's stack before moving it.
        alloc::boxed::Box::new(Self {
            candidate_w1: [[0.0; HIDDEN_DIM]; CANDIDATE_INPUT_DIM],
            candidate_b1: [0.0; HIDDEN_DIM],
            trunk_w: [[0.0; HIDDEN_DIM]; HIDDEN_DIM],
            trunk_b: [0.0; HIDDEN_DIM],
            candidate_proj_w: [[0.0; EMBED_DIM]; HIDDEN_DIM],
            candidate_proj_b: [0.0; EMBED_DIM],
            mask_mlp_w1: [[0.0; MLP_HIDDEN]; EMBED_DIM],
            mask_mlp_b1: [0.0; MLP_HIDDEN],
            mask_mlp_w2: [[0.0; EMBED_DIM]; MLP_HIDDEN],
            mask_mlp_b2: [0.0; EMBED_DIM],
            interaction: [[0.0; EMBED_DIM]; EMBED_DIM],
            mask_bias_proj: [0.0; EMBED_DIM],
            rule_proj_w: [[0.0; EMBED_DIM]; RULE_CONCAT_DIM],
            rule_proj_b: [0.0; EMBED_DIM],
        })
    }

    /// Set every accumulated gradient back to zero, in place.
    pub(crate) fn clear(&mut self) {
        self.candidate_w1 = [[0.0; HIDDEN_DIM]; CANDIDATE_INPUT_DIM];
        self.candidate_b1 = [0.0; HIDDEN_DIM];
        self.trunk_w = [[0.0; HIDDEN_DIM]; HIDDEN_DIM];
        self.trunk_b = [0.0; HIDDEN_DIM];
        self.candidate_proj_w = [[0.0; EMBED_DIM]; HIDDEN_DIM];
        self.candidate_proj_b = [0.0; EMBED_DIM];
        self.mask_mlp_w1 = [[0.0; MLP_HIDDEN]; EMBED_DIM];
        self.mask_mlp_b1 = [0.0; MLP_HIDDEN];
        self.mask_mlp_w2 = [[0.0; EMBED_DIM]; MLP_HIDDEN];
        self.mask_mlp_b2 = [0.0; EMBED_DIM];
        self.interaction = [[0.0; EMBED_DIM]; EMBED_DIM];
        self.mask_bias_proj = [0.0; EMBED_DIM];
        self.rule_proj_w = [[0.0; EMBED_DIM]; RULE_CONCAT_DIM];
        self.rule_proj_b = [0.0; EMBED_DIM];
    }

    /// Euclidean norm over every accumulated gradient.
    ///
    /// Also the cheapest honest liveness check this accumulator has: a
    /// non-finite norm is a diverged step, and it is the one number a caller
    /// already needs (for clipping) rather than a 15,000-element scan added
    /// for the sake of an assertion.
    #[must_use]
    pub(crate) fn l2_norm(&self) -> f32 {
        let mut sum = 0.0f64;
        let mut add = |v: f32| sum += f64::from(v) * f64::from(v);
        for row in &self.candidate_w1 {
            for v in row {
                add(*v);
            }
        }
        for v in &self.candidate_b1 {
            add(*v);
        }
        for row in &self.trunk_w {
            for v in row {
                add(*v);
            }
        }
        for v in &self.trunk_b {
            add(*v);
        }
        for row in &self.candidate_proj_w {
            for v in row {
                add(*v);
            }
        }
        for v in &self.candidate_proj_b {
            add(*v);
        }
        for row in &self.mask_mlp_w1 {
            for v in row {
                add(*v);
            }
        }
        for v in &self.mask_mlp_b1 {
            add(*v);
        }
        for row in &self.mask_mlp_w2 {
            for v in row {
                add(*v);
            }
        }
        for v in &self.mask_mlp_b2 {
            add(*v);
        }
        for row in &self.interaction {
            for v in row {
                add(*v);
            }
        }
        for v in &self.mask_bias_proj {
            add(*v);
        }
        for row in &self.rule_proj_w {
            for v in row {
                add(*v);
            }
        }
        for v in &self.rule_proj_b {
            add(*v);
        }
        libm::sqrt(sum) as f32
    }

    /// Multiply every accumulated gradient by `factor`.
    pub(crate) fn scale(&mut self, factor: f32) {
        let f = |v: &mut f32| *v *= factor;
        for row in &mut self.candidate_w1 {
            for v in row {
                f(v);
            }
        }
        for v in &mut self.candidate_b1 {
            f(v);
        }
        for row in &mut self.trunk_w {
            for v in row {
                f(v);
            }
        }
        for v in &mut self.trunk_b {
            f(v);
        }
        for row in &mut self.candidate_proj_w {
            for v in row {
                f(v);
            }
        }
        for v in &mut self.candidate_proj_b {
            f(v);
        }
        for row in &mut self.mask_mlp_w1 {
            for v in row {
                f(v);
            }
        }
        for v in &mut self.mask_mlp_b1 {
            f(v);
        }
        for row in &mut self.mask_mlp_w2 {
            for v in row {
                f(v);
            }
        }
        for v in &mut self.mask_mlp_b2 {
            f(v);
        }
        for row in &mut self.interaction {
            for v in row {
                f(v);
            }
        }
        for v in &mut self.mask_bias_proj {
            f(v);
        }
        for row in &mut self.rule_proj_w {
            for v in row {
                f(v);
            }
        }
        for v in &mut self.rule_proj_b {
            f(v);
        }
    }

    /// Chain a rule-embedding gradient (`dL/dr`, as
    /// [`SaturationHead::backward`] returns it) back through
    /// [`SaturationHead::project_rule`] onto the rule projection's own
    /// weights.
    ///
    /// Separate from `backward` because the rule *encoder*'s input — the
    /// `[LHS | RHS | LHS−RHS | LHS⊙RHS]` concatenation — is a property of
    /// the rule, computed once per rule per epoch, not of the candidate.
    /// Whether it is called at all is the "do we train the rule embedding"
    /// decision, made explicitly at the call site rather than hidden inside
    /// the candidate backward pass.
    pub(crate) fn accumulate_rule_proj(
        &mut self,
        concat: &[f32; RULE_CONCAT_DIM],
        d_rule_embed: &[f32; EMBED_DIM],
    ) {
        for k in 0..EMBED_DIM {
            self.rule_proj_b[k] += d_rule_embed[k];
        }
        for i in 0..RULE_CONCAT_DIM {
            let c = concat[i];
            if c == 0.0 {
                continue;
            }
            for k in 0..EMBED_DIM {
                self.rule_proj_w[i][k] += c * d_rule_embed[k];
            }
        }
    }
}

impl SaturationHead {
    /// Accumulate `d_score * dscore/dw` into `grads` for every weight on the
    /// candidate-scoring path, and return `dscore/dr * d_score` — the
    /// gradient with respect to the rule embedding, for a caller that wants
    /// to chain it into [`HeadGradients::accumulate_rule_proj`].
    ///
    /// `d_score` is `dLoss/dscore`; this function knows nothing about the
    /// loss.
    pub(crate) fn backward(
        &self,
        acts: &Activations,
        d_score: f32,
        grads: &mut HeadGradients,
    ) -> [f32; EMBED_DIM] {
        // score = Σ_k (transformed[k] + mask_bias_proj[k]) * rule_embed[k]
        let mut d_transformed = [0.0f32; EMBED_DIM];
        let mut d_rule_embed = [0.0f32; EMBED_DIM];
        for k in 0..EMBED_DIM {
            d_transformed[k] = d_score * acts.rule_embed[k];
            grads.mask_bias_proj[k] += d_score * acts.rule_embed[k];
            d_rule_embed[k] = d_score * (acts.transformed[k] + self.mask_bias_proj[k]);
        }

        // transformed[j] = Σ_i mask_features[i] * interaction[i][j]
        let mut d_mask = [0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            let mi = acts.mask_features[i];
            let mut acc = 0.0f32;
            for j in 0..EMBED_DIM {
                grads.interaction[i][j] += mi * d_transformed[j];
                acc += self.interaction[i][j] * d_transformed[j];
            }
            d_mask[i] = acc;
        }

        // mask_features[k] = mask_mlp_b2[k] + Σ_j g[j] * mask_mlp_w2[j][k]
        for k in 0..EMBED_DIM {
            grads.mask_mlp_b2[k] += d_mask[k];
        }
        let mut d_g = [0.0f32; MLP_HIDDEN];
        for j in 0..MLP_HIDDEN {
            let gj = acts.g[j];
            let mut acc = 0.0f32;
            for k in 0..EMBED_DIM {
                grads.mask_mlp_w2[j][k] += gj * d_mask[k];
                acc += self.mask_mlp_w2[j][k] * d_mask[k];
            }
            // ReLU: the pre-activation was positive exactly when g[j] > 0.
            d_g[j] = if gj > 0.0 { acc } else { 0.0 };
        }

        // g = relu(mask_mlp_b1 + embedᵀ mask_mlp_w1)
        for j in 0..MLP_HIDDEN {
            grads.mask_mlp_b1[j] += d_g[j];
        }
        let mut d_embed = [0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            let ei = acts.embed[i];
            let mut acc = 0.0f32;
            for j in 0..MLP_HIDDEN {
                grads.mask_mlp_w1[i][j] += ei * d_g[j];
                acc += self.mask_mlp_w1[i][j] * d_g[j];
            }
            d_embed[i] = acc;
        }

        // embed = candidate_proj_b + h2ᵀ candidate_proj_w
        for k in 0..EMBED_DIM {
            grads.candidate_proj_b[k] += d_embed[k];
        }
        let mut d_h2 = [0.0f32; HIDDEN_DIM];
        for j in 0..HIDDEN_DIM {
            let hj = acts.h2[j];
            let mut acc = 0.0f32;
            for k in 0..EMBED_DIM {
                grads.candidate_proj_w[j][k] += hj * d_embed[k];
                acc += self.candidate_proj_w[j][k] * d_embed[k];
            }
            d_h2[j] = if hj > 0.0 { acc } else { 0.0 };
        }

        // h2 = relu(trunk_b + h1ᵀ trunk_w)
        for j in 0..HIDDEN_DIM {
            grads.trunk_b[j] += d_h2[j];
        }
        let mut d_h1 = [0.0f32; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            let hi = acts.h1[i];
            let mut acc = 0.0f32;
            for j in 0..HIDDEN_DIM {
                grads.trunk_w[i][j] += hi * d_h2[j];
                acc += self.trunk_w[i][j] * d_h2[j];
            }
            d_h1[i] = if hi > 0.0 { acc } else { 0.0 };
        }

        // h1 = relu(candidate_b1 + inputᵀ candidate_w1); `input` is a
        // function of the FROZEN op embeddings, so the chain stops here.
        for j in 0..HIDDEN_DIM {
            grads.candidate_b1[j] += d_h1[j];
        }
        for i in 0..CANDIDATE_INPUT_DIM {
            let x = acts.input[i];
            if x == 0.0 {
                continue;
            }
            for j in 0..HIDDEN_DIM {
                grads.candidate_w1[i][j] += x * d_h1[j];
            }
        }

        d_rule_embed
    }

    /// Set every ReLU bias on the candidate-scoring path to `bias`.
    ///
    /// A ReLU unit whose pre-activation is negative for every input in the
    /// data is **dead**: it passes no gradient, so it never comes back. Dead
    /// units in the *mask MLP* are a special hazard for this head rather
    /// than a generic training nuisance — if all `MLP_HIDDEN` of them close,
    /// `mask_features` collapses to the constant `mask_mlp_b2`, the bilinear
    /// score becomes `(constᵀW + bᵀ)r`, and the model **is** additively
    /// separable. The head would then be silently measuring the very model
    /// class the registered comparison is trying to distinguish it from.
    ///
    /// `SaturationHead::randomize`'s He initialisation leaves only about a
    /// third of the mask MLP open at realistic input scales, so training
    /// starts one step away from that collapse. This offsets every ReLU
    /// bias positive instead. It is an initialisation, not a constraint:
    /// training is free to close a unit that should be closed.
    pub(crate) fn warm_relu_biases(&mut self, bias: f32) {
        self.candidate_b1 = [bias; HIDDEN_DIM];
        self.trunk_b = [bias; HIDDEN_DIM];
        self.mask_mlp_b1 = [bias; MLP_HIDDEN];
    }

    /// Visit every **trained** parameter, in the canonical checkpoint order.
    ///
    /// One definition of that order, used by
    /// [`Self::trained_parameters`], [`Self::load_trained_parameters`] and
    /// [`Self::trained_parameter_count`] alike: a save routine and a load
    /// routine that each spell the order out separately are two orders, and
    /// the day they disagree the checkpoint still loads — as a transposed,
    /// silently wrong model.
    ///
    /// The set is exactly [`HeadGradients`]'s tensors: the `graph_*` tower
    /// is not a trained parameter of this head (nothing on the
    /// candidate-scoring path reads it), so it is not in a checkpoint of
    /// one.
    fn for_each_trained_parameter(&mut self, f: &mut impl FnMut(&mut f32)) {
        for row in &mut self.candidate_w1 {
            for w in row {
                f(w);
            }
        }
        for b in &mut self.candidate_b1 {
            f(b);
        }
        for row in &mut self.trunk_w {
            for w in row {
                f(w);
            }
        }
        for b in &mut self.trunk_b {
            f(b);
        }
        for row in &mut self.candidate_proj_w {
            for w in row {
                f(w);
            }
        }
        for b in &mut self.candidate_proj_b {
            f(b);
        }
        for row in &mut self.mask_mlp_w1 {
            for w in row {
                f(w);
            }
        }
        for b in &mut self.mask_mlp_b1 {
            f(b);
        }
        for row in &mut self.mask_mlp_w2 {
            for w in row {
                f(w);
            }
        }
        for b in &mut self.mask_mlp_b2 {
            f(b);
        }
        for row in &mut self.interaction {
            for w in row {
                f(w);
            }
        }
        for b in &mut self.mask_bias_proj {
            f(b);
        }
        for row in &mut self.rule_proj_w {
            for w in row {
                f(w);
            }
        }
        for b in &mut self.rule_proj_b {
            f(b);
        }
    }

    /// How many floats [`Self::trained_parameters`] produces.
    #[must_use]
    pub(crate) fn trained_parameter_count() -> usize {
        CANDIDATE_INPUT_DIM * HIDDEN_DIM
            + HIDDEN_DIM
            + HIDDEN_DIM * HIDDEN_DIM
            + HIDDEN_DIM
            + HIDDEN_DIM * EMBED_DIM
            + EMBED_DIM
            + EMBED_DIM * MLP_HIDDEN
            + MLP_HIDDEN
            + MLP_HIDDEN * EMBED_DIM
            + EMBED_DIM
            + EMBED_DIM * EMBED_DIM
            + EMBED_DIM
            + RULE_CONCAT_DIM * EMBED_DIM
            + EMBED_DIM
    }

    /// This head's trained parameters, flat, in the canonical order.
    #[must_use]
    pub(crate) fn trained_parameters(&self) -> alloc::vec::Vec<f32> {
        let mut out = alloc::vec::Vec::with_capacity(Self::trained_parameter_count());
        // `for_each_trained_parameter` needs `&mut self` to hand out `&mut
        // f32` slots; a clone keeps this method's `&self` honest and costs
        // one head-sized copy per checkpoint written.
        let mut scratch = self.clone();
        scratch.for_each_trained_parameter(&mut |w| out.push(*w));
        out
    }

    /// Overwrite this head's trained parameters from a flat vector in the
    /// canonical order.
    ///
    /// # Panics
    ///
    /// If `params` is not exactly [`Self::trained_parameter_count`] long.
    /// The caller (`bilinear::BilinearCandidateGuide::new`) checks and
    /// reports the length first; this is the backstop for anything that
    /// does not.
    pub(crate) fn load_trained_parameters(&mut self, params: &[f32]) {
        assert_eq!(
            params.len(),
            Self::trained_parameter_count(),
            "SaturationHead::load_trained_parameters: wrong parameter count"
        );
        let mut i = 0usize;
        self.for_each_trained_parameter(&mut |w| {
            *w = params[i];
            i += 1;
        });
        assert_eq!(
            i,
            params.len(),
            "parameter visitor consumed the wrong count"
        );
    }

    /// Apply one SGD step: `w -= lr * (grad + l2 * w)` for every weight,
    /// `w -= lr * grad` for every bias.
    ///
    /// L2 decays **weights only**, and decays every weight rather than only
    /// the ones this step's sample touched — the same discipline
    /// `guide_linear::Model::sgd_step` is pinned to by
    /// `l2_decays_every_weight_not_only_the_ones_a_sample_touched`, for the
    /// same reason: "decay what the gradient touched" is a different
    /// regularizer that depends on the data order.
    pub(crate) fn apply_sgd(&mut self, grads: &HeadGradients, lr: f32, l2: f32) {
        #[inline]
        fn step_w(w: &mut f32, g: f32, lr: f32, l2: f32) {
            *w -= lr * (g + l2 * *w);
        }
        #[inline]
        fn step_b(b: &mut f32, g: f32, lr: f32) {
            *b -= lr * g;
        }

        for i in 0..CANDIDATE_INPUT_DIM {
            for j in 0..HIDDEN_DIM {
                step_w(
                    &mut self.candidate_w1[i][j],
                    grads.candidate_w1[i][j],
                    lr,
                    l2,
                );
            }
        }
        for j in 0..HIDDEN_DIM {
            step_b(&mut self.candidate_b1[j], grads.candidate_b1[j], lr);
            step_b(&mut self.trunk_b[j], grads.trunk_b[j], lr);
        }
        for i in 0..HIDDEN_DIM {
            for j in 0..HIDDEN_DIM {
                step_w(&mut self.trunk_w[i][j], grads.trunk_w[i][j], lr, l2);
            }
        }
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                step_w(
                    &mut self.candidate_proj_w[j][k],
                    grads.candidate_proj_w[j][k],
                    lr,
                    l2,
                );
            }
        }
        for k in 0..EMBED_DIM {
            step_b(&mut self.candidate_proj_b[k], grads.candidate_proj_b[k], lr);
            step_b(&mut self.mask_mlp_b2[k], grads.mask_mlp_b2[k], lr);
            step_b(&mut self.rule_proj_b[k], grads.rule_proj_b[k], lr);
            step_w(&mut self.mask_bias_proj[k], grads.mask_bias_proj[k], lr, l2);
        }
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                step_w(&mut self.mask_mlp_w1[i][j], grads.mask_mlp_w1[i][j], lr, l2);
            }
        }
        for j in 0..MLP_HIDDEN {
            step_b(&mut self.mask_mlp_b1[j], grads.mask_mlp_b1[j], lr);
            for k in 0..EMBED_DIM {
                step_w(&mut self.mask_mlp_w2[j][k], grads.mask_mlp_w2[j][k], lr, l2);
            }
        }
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                step_w(&mut self.interaction[i][j], grads.interaction[i][j], lr, l2);
            }
        }
        for i in 0..RULE_CONCAT_DIM {
            for k in 0..EMBED_DIM {
                step_w(&mut self.rule_proj_w[i][k], grads.rule_proj_w[i][k], lr, l2);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::rules::RuleId;
    use crate::nnue::factored::K;
    use crate::nnue::guide::scoring::{RULE_CONCAT_DIM, rule_concat};
    use alloc::vec;
    use pixelflow_ir::OpKind;

    /// Central-difference step, and the two-sided derivation that fixes it.
    ///
    /// The error of `(f(w+h) − f(w−h)) / 2h` evaluated in `f32` has two terms
    /// that trade against each other:
    ///
    /// ```text
    /// err(h)  ≈  C·ε·|f| / h       roundoff:   each evaluation is accurate to
    ///                                          ~C·ε·|f|; the subtraction cancels
    ///                                          the signal and the /2h amplifies
    ///                                          whatever survives
    ///         +  (h²/6)·|f‴|       truncation: the Taylor remainder
    /// ```
    ///
    /// with `ε = f32::EPSILON`. For a *general* `f` the two balance at
    /// `h* ≈ (3·C·ε·|f| / |f‴|)^(1/3)` and nothing does better than the error
    /// there. This network is not general: it is a ReLU MLP followed by a
    /// bilinear form, so as a function of any **single** weight, and inside one
    /// activation pattern, the score is **affine**. `f‴ ≡ 0` — the truncation
    /// term is absent, and it is replaced by a hard wall instead of a curve.
    /// So `H` is squeezed between two bounds rather than balanced at an
    /// optimum, and both are asserted rather than asserted-in-a-comment:
    ///
    /// * **Below**, by roundoff. `ε·|score| / H` is an absolute error on the
    ///   estimate, against a budget of `TOL · max(1, |numeric|)` — i.e. `TOL`
    ///   at worst. Requiring [`FLOOR_MARGIN`] of headroom gives
    ///   `H > 16 × 1.1921e-7 × 234.8 / 2e-2 = 2.2e-2`.
    ///   [`the_step_size_must_keep_the_roundoff_floor_under_the_tolerance`]
    ///   re-derives this from the fixture's own measured score.
    /// * **Above**, by the activation cell. A step that opens or closes a ReLU
    ///   differences across two affine pieces and estimates neither slope. The
    ///   fixture's narrowest cell is `≈ 8.2e-2` (at `candidate_w1[35][*]` —
    ///   scalar row 35 is `ln(1 + expr_node_count) ≈ 3.18`, the largest input,
    ///   so it moves the tower furthest per unit of weight). [`check_gradient`]
    ///   asserts on **every** probe that the pattern is unchanged at `w ± H`.
    ///
    /// `4e-2` sits inside `(2.2e-2, 8.2e-2)` with roughly balanced margin:
    /// 1.8× above the roundoff bound (a 29× floor-to-`TOL` ratio) and 2.1×
    /// below the kink bound. Measured error at that step is ~1e-3, twenty times
    /// under `TOL`, on every build.
    ///
    /// This constant originally held `1e-3`, chosen as if the truncation term
    /// existed. The roundoff floor there is `1.1921e-7 × 234.8 / 1e-3 = 2.8e-2`
    /// — **1.4× above [`TOL`]**. The check demanded more precision than an
    /// `f32` central difference can deliver on any target, so whether it passed
    /// was decided by which way the rounding fell: green at the baseline ISA,
    /// red at avx2+fma on `candidate_w1[5][7]` (postsubmit on #1124 and #1128),
    /// red again at SSE2 on other hardware. It was never an ISA divergence; it
    /// was a check with no error budget.
    const H: f32 = 4e-2;
    /// Relative tolerance for `|analytic − numeric| / max(1, |numeric|)`.
    const TOL: f32 = 2e-2;
    /// How far under [`TOL`] the central difference's roundoff floor
    /// `eps·|score|/H` must sit for the comparison to be measuring the gradient
    /// rather than the noise. It absorbs the accumulation factor `C` the floor
    /// formula writes as 1: six chained dot products put `C` near 2.5 in the
    /// failures this replaces, so 16 leaves 6x over the worst `C` observed —
    /// and the fixture's actual margin is 71x, so this is a bound, not a fit.
    /// Held by [`the_step_size_must_keep_the_roundoff_floor_under_the_tolerance`].
    const FLOOR_MARGIN: f32 = 16.0;

    /// A head whose every tensor is filled from an index-dependent,
    /// **asymmetric** formula: `w[i][j]` depends on `i` and `j` differently,
    /// so a transposed index produces a different number rather than the
    /// same one. `SaturationHead::randomize` would also do this, but a
    /// closed-form fixture makes a failure readable.
    fn asymmetric_head() -> SaturationHead {
        let mut head = SaturationHead::new();
        let f = |i: usize, j: usize, salt: usize| -> f32 {
            let x = (i as f32 * 0.37 + j as f32 * 0.11 + salt as f32 * 0.53).sin();
            x * 0.4
        };
        // Biases are positive and bounded away from zero so every ReLU on
        // the path is *open* at the probe point. With small zero-mean
        // biases the mask MLP's 16 hidden units all closed, `d_g` was
        // identically zero, and every gradient below it was zero too — the
        // finite-difference checks then compared 0 against 0 and passed
        // while testing nothing. `every_relu_on_the_path_should_be_open_at_
        // the_fixture_point` pins that this is still true.
        let bias = |j: usize, salt: usize| -> f32 { f(j, 0, salt).abs() + 0.25 };
        for i in 0..CANDIDATE_INPUT_DIM {
            for j in 0..HIDDEN_DIM {
                head.candidate_w1[i][j] = f(i, j, 1);
            }
        }
        for (j, b) in head.candidate_b1.iter_mut().enumerate() {
            *b = bias(j, 2);
        }
        for i in 0..HIDDEN_DIM {
            for j in 0..HIDDEN_DIM {
                head.trunk_w[i][j] = f(i, j, 3);
            }
        }
        for (j, b) in head.trunk_b.iter_mut().enumerate() {
            *b = bias(j, 4);
        }
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                head.candidate_proj_w[j][k] = f(j, k, 5);
            }
        }
        for (k, b) in head.candidate_proj_b.iter_mut().enumerate() {
            *b = f(k, 0, 6);
        }
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                head.mask_mlp_w1[i][j] = f(i, j, 7);
            }
        }
        for (j, b) in head.mask_mlp_b1.iter_mut().enumerate() {
            *b = bias(j, 8);
        }
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                head.mask_mlp_w2[j][k] = f(j, k, 9);
            }
        }
        for (k, b) in head.mask_mlp_b2.iter_mut().enumerate() {
            *b = f(k, 0, 10);
        }
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                head.interaction[i][j] = f(i, j, 11);
            }
        }
        for (k, b) in head.mask_bias_proj.iter_mut().enumerate() {
            *b = f(k, 0, 12);
        }
        for i in 0..RULE_CONCAT_DIM {
            for k in 0..EMBED_DIM {
                head.rule_proj_w[i][k] = f(i, k, 13);
            }
        }
        for (k, b) in head.rule_proj_b.iter_mut().enumerate() {
            *b = f(k, 0, 14);
        }
        head
    }

    fn embeddings() -> OpEmbeddings {
        OpEmbeddings::new_random(7)
    }

    /// Two side embeddings with distinct, non-degenerate content: the
    /// concatenation's four blocks (`LHS`, `RHS`, `LHS−RHS`, `LHS⊙RHS`) must
    /// all be nonzero, or a `rule_proj_w` gradient check would pass over
    /// rows that never fire.
    fn sides() -> ([f32; EMBED_DIM], [f32; EMBED_DIM]) {
        let mut l = [0.0f32; EMBED_DIM];
        let mut r = [0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            l[i] = ((i as f32) * 0.23 + 0.4).cos() * 0.8;
            r[i] = ((i as f32) * 0.17 - 0.3).sin() * 0.7 + 0.2;
        }
        (l, r)
    }

    /// A candidate whose neighborhood, budget and both size scalars are all
    /// nonzero and all different — every input row live, so no gradient
    /// check silently passes on a zero.
    fn candidate(head: &SaturationHead) -> CandidateSummary {
        let (l, r) = sides();
        CandidateSummary {
            rule_embed: head.encode_rule(&l, &r),
            neighborhood_ops: vec![OpKind::Sin, OpKind::Add, OpKind::Sin, OpKind::Sqrt],
            budget_fraction: 0.37,
            rule: RuleId::from_label("gradient-check"),
            match_class_node_count: 5,
            expr_node_count: 23,
        }
    }

    /// Analytic gradients of the score (i.e. `d_score = 1`) w.r.t. every
    /// trained tensor, including the rule projection.
    fn analytic() -> (SaturationHead, alloc::boxed::Box<HeadGradients>) {
        let head = asymmetric_head();
        let emb = embeddings();
        let c = candidate(&head);
        let acts = Activations::forward(&head, &emb, &c);
        let mut grads = HeadGradients::zero();
        let d_rule = head.backward(&acts, 1.0, &mut grads);
        let (l, r) = sides();
        grads.accumulate_rule_proj(&rule_concat(&l, &r), &d_rule);
        (head, grads)
    }

    /// The fixture's ReLU pattern — which hidden units are open — one bit per
    /// unit, for `h1`, `h2` and `g`.
    ///
    /// Inside a single pattern the score is affine in any single weight, which
    /// is the entire justification for [`H`]'s size. A probe that changes the
    /// pattern has crossed a kink.
    fn relu_pattern(head: &SaturationHead, emb: &OpEmbeddings, c: &CandidateSummary) -> [u64; 3] {
        let acts = Activations::forward(head, emb, &rescored(head, c));
        let bits = |units: &[f32]| {
            units
                .iter()
                .enumerate()
                .fold(0u64, |m, (i, &v)| if v > 0.0 { m | (1 << i) } else { m })
        };
        [bits(&acts.h1), bits(&acts.h2), bits(&acts.g)]
    }

    /// `c` with its rule embedding recomputed from `head`, so a perturbation of
    /// `rule_proj_*` is visible to whatever consumes the result.
    fn rescored(head: &SaturationHead, c: &CandidateSummary) -> CandidateSummary {
        let (l, r) = sides();
        CandidateSummary {
            rule_embed: head.encode_rule(&l, &r),
            neighborhood_ops: c.neighborhood_ops.clone(),
            budget_fraction: c.budget_fraction,
            rule: c.rule,
            match_class_node_count: c.match_class_node_count,
            expr_node_count: c.expr_node_count,
        }
    }

    /// Forward score, recomputing the rule embedding from `head` so a
    /// perturbation of `rule_proj_*` is visible.
    fn score_of(head: &SaturationHead, emb: &OpEmbeddings, c: &CandidateSummary) -> f32 {
        head.score_candidate(emb, &rescored(head, c))
    }

    /// Check one weight's analytic gradient against the central difference
    /// `(f(w+H) − f(w−H)) / 2H`.
    ///
    /// The kink assertion is not decoration. [`H`] is large precisely because
    /// the score is affine in a single weight *within one activation pattern*;
    /// a probe that opens or closes a ReLU differences across two affine
    /// pieces, and its estimate is a chord between two slopes rather than
    /// either of them. That must fail loudly, naming the cause, instead of
    /// arriving later as an unexplained gradient disagreement.
    fn check_gradient(
        head: &SaturationHead,
        name: &str,
        analytic: f32,
        set: impl Fn(&mut SaturationHead, f32),
    ) {
        let emb = embeddings();
        let c = candidate(head);
        let mut plus = head.clone();
        set(&mut plus, H);
        let mut minus = head.clone();
        set(&mut minus, -H);

        let at = relu_pattern(head, &emb, &c);
        let up = relu_pattern(&plus, &emb, &c);
        let down = relu_pattern(&minus, &emb, &c);
        assert!(
            up == at && down == at,
            "{name}: the ±H probe crossed a ReLU kink (open-unit masks h1/h2/g \
             = {at:x?} at w, {up:x?} at w+H, {down:x?} at w−H). The central \
             difference then straddles two affine pieces and estimates neither \
             slope, so H = {H} is too large for this fixture — it is not a \
             gradient disagreement."
        );

        let numeric = (score_of(&plus, &emb, &c) - score_of(&minus, &emb, &c)) / (2.0 * H);
        let denom = numeric.abs().max(1.0);
        assert!(
            (analytic - numeric).abs() / denom < TOL,
            "{name}: analytic {analytic} vs numeric {numeric}"
        );
    }

    #[test]
    fn the_step_size_must_keep_the_roundoff_floor_under_the_tolerance() {
        // The shift-left, and the guard that would have made #1124's postsubmit
        // failure impossible to write. Every check in this module compares an
        // analytic gradient against an `f32` central difference whose roundoff
        // floor is `C·eps·|score| / H` — an absolute error on the *estimate*,
        // against a budget of `TOL · max(1, |numeric|)`, i.e. `TOL` at worst.
        // When the floor is not comfortably under that budget, the comparison
        // measures rounding rather than the gradient and its verdict is decided
        // per target: at H = 1e-3 the floor was 1.4x *over* TOL and
        // `candidate_w1[5][7]` went red at avx2+fma while passing at the
        // baseline ISA.
        //
        // Deriving the floor here from the fixture's own score is what makes
        // the next out-of-budget check fail at authoring time instead of
        // intermittently on one ISA in postsubmit: grow the fixture's
        // magnitude, or shrink H, and this fails first, with the number.
        let head = asymmetric_head();
        let emb = embeddings();
        let c = candidate(&head);
        let score = head.score_candidate(&emb, &c).abs();
        let floor = f32::EPSILON * score / H;
        assert!(
            floor * FLOOR_MARGIN < TOL,
            "the central difference's roundoff floor eps*|score|/H = {floor:e} \
             (|score| = {score}, H = {H}) is not {FLOOR_MARGIN}x under TOL = \
             {TOL:e}, so the gradient checks in this module would be measuring \
             rounding, not the gradient. Raise H — the score is piecewise \
             affine in a single weight, so a larger step costs no truncation, \
             only a wider activation cell to stay inside — or shrink the \
             fixture's magnitude."
        );
    }

    #[test]
    fn activations_score_should_equal_score_candidate() {
        // The two paths compute the same score by separately written code, so
        // they agree up to summation order — not bit-for-bit. The bound is
        // therefore relative and derived: the chain's longest single reduction
        // is `HIDDEN_DIM` terms, whose accumulated rounding is at worst
        // `HIDDEN_DIM · eps · |score|`.
        //
        // This assertion previously read `< 1e-5` absolute, which at
        // |score| = 234.8 is *below one ulp* (1.37e-5) — it demanded bit
        // identity from two different summation orders. It held only because
        // CI builds tests at opt-level 0; at opt-level 3 on aarch64 the two
        // paths land one ulp apart and it fails. Same defect as `H`'s, in the
        // same file: a tolerance with no error model behind it.
        let head = asymmetric_head();
        let emb = embeddings();
        let c = candidate(&head);
        let acts = Activations::forward(&head, &emb, &c);
        let direct = head.score_candidate(&emb, &c);
        let bound = HIDDEN_DIM as f32 * f32::EPSILON * direct.abs().max(1.0);
        assert!(
            (acts.score() - direct).abs() <= bound,
            "the backward pass must be differentiating the function \
             `score_candidate` computes: {} vs {direct} (bound {bound:e})",
            acts.score()
        );
    }

    #[test]
    fn every_relu_on_the_path_should_be_open_at_the_fixture_point() {
        // The guard on every other check in this module: a fixture whose
        // ReLUs are all closed makes `analytic()` return all-zero gradients,
        // and a central difference of a locally-constant function is also
        // zero, so every `check_gradient` would pass against a dead network.
        let head = asymmetric_head();
        let emb = embeddings();
        let c = candidate(&head);
        let acts = Activations::forward(&head, &emb, &c);
        assert!(
            acts.h1.iter().any(|&v| v > 0.0),
            "candidate tower ReLU fully closed"
        );
        assert!(acts.h2.iter().any(|&v| v > 0.0), "trunk ReLU fully closed");
        assert!(
            acts.g.iter().any(|&v| v > 0.0),
            "mask MLP ReLU fully closed"
        );

        let mut grads = HeadGradients::zero();
        let _ = head.backward(&acts, 1.0, &mut grads);
        let live = grads
            .candidate_w1
            .iter()
            .flatten()
            .filter(|v| v.abs() > 0.0)
            .count();
        assert!(
            live > 0,
            "the candidate tower must receive a nonzero gradient at the fixture point"
        );
    }

    #[test]
    fn numerical_gradient_check_interaction_matrix() {
        // The transpose trap: `interaction` is square, so a swapped index is
        // well-typed. Off-diagonal entries with i != j are the only ones
        // that can see the difference, so the probes are all off-diagonal
        // and asymmetric (both (i,j) and (j,i) are checked).
        let (head, grads) = analytic();
        for &(i, j) in &[(0usize, 5usize), (5, 0), (3, 17), (17, 3), (31, 2)] {
            check_gradient(
                &head,
                &alloc::format!("interaction[{i}][{j}]"),
                grads.interaction[i][j],
                |h, d| h.interaction[i][j] += d,
            );
        }
    }

    #[test]
    fn numerical_gradient_check_mask_bias_proj() {
        let (head, grads) = analytic();
        for &k in &[0usize, 7, 31] {
            check_gradient(
                &head,
                &alloc::format!("mask_bias_proj[{k}]"),
                grads.mask_bias_proj[k],
                |h, d| h.mask_bias_proj[k] += d,
            );
        }
    }

    #[test]
    fn numerical_gradient_check_rule_projection() {
        // The rule embedding's gradient path: score -> r -> rule_proj. A
        // sign error here trains the rule encoder backwards, which no loss
        // curve distinguishes from a hard problem.
        let (head, grads) = analytic();
        for &(i, k) in &[(0usize, 0usize), (3, 11), (40, 5), (70, 29), (127, 31)] {
            check_gradient(
                &head,
                &alloc::format!("rule_proj_w[{i}][{k}]"),
                grads.rule_proj_w[i][k],
                |h, d| h.rule_proj_w[i][k] += d,
            );
        }
        for &k in &[0usize, 13, 31] {
            check_gradient(
                &head,
                &alloc::format!("rule_proj_b[{k}]"),
                grads.rule_proj_b[k],
                |h, d| h.rule_proj_b[k] += d,
            );
        }
    }

    #[test]
    fn numerical_gradient_check_mask_mlp() {
        let (head, grads) = analytic();
        for &(i, j) in &[(0usize, 0usize), (4, 9), (31, 2)] {
            check_gradient(
                &head,
                &alloc::format!("mask_mlp_w1[{i}][{j}]"),
                grads.mask_mlp_w1[i][j],
                |h, d| h.mask_mlp_w1[i][j] += d,
            );
        }
        for &(j, k) in &[(0usize, 0usize), (7, 21), (15, 31)] {
            check_gradient(
                &head,
                &alloc::format!("mask_mlp_w2[{j}][{k}]"),
                grads.mask_mlp_w2[j][k],
                |h, d| h.mask_mlp_w2[j][k] += d,
            );
        }
        for &j in &[0usize, 9, 15] {
            check_gradient(
                &head,
                &alloc::format!("mask_mlp_b1[{j}]"),
                grads.mask_mlp_b1[j],
                |h, d| h.mask_mlp_b1[j] += d,
            );
        }
        for &k in &[0usize, 20, 31] {
            check_gradient(
                &head,
                &alloc::format!("mask_mlp_b2[{k}]"),
                grads.mask_mlp_b2[k],
                |h, d| h.mask_mlp_b2[k] += d,
            );
        }
    }

    #[test]
    fn numerical_gradient_check_candidate_projection_and_trunk() {
        let (head, grads) = analytic();
        for &(j, k) in &[(0usize, 0usize), (13, 6), (63, 31)] {
            check_gradient(
                &head,
                &alloc::format!("candidate_proj_w[{j}][{k}]"),
                grads.candidate_proj_w[j][k],
                |h, d| h.candidate_proj_w[j][k] += d,
            );
        }
        for &k in &[0usize, 17, 31] {
            check_gradient(
                &head,
                &alloc::format!("candidate_proj_b[{k}]"),
                grads.candidate_proj_b[k],
                |h, d| h.candidate_proj_b[k] += d,
            );
        }
        // trunk_w is square too — same transpose trap as `interaction`.
        for &(i, j) in &[(0usize, 11usize), (11, 0), (40, 63), (63, 40)] {
            check_gradient(
                &head,
                &alloc::format!("trunk_w[{i}][{j}]"),
                grads.trunk_w[i][j],
                |h, d| h.trunk_w[i][j] += d,
            );
        }
        for &j in &[0usize, 33, 63] {
            check_gradient(
                &head,
                &alloc::format!("trunk_b[{j}]"),
                grads.trunk_b[j],
                |h, d| h.trunk_b[j] += d,
            );
        }
    }

    #[test]
    fn numerical_gradient_check_candidate_tower_including_every_scalar_row() {
        let (head, grads) = analytic();
        // Pooled-op rows...
        for &(i, j) in &[(0usize, 0usize), (5, 7), (31, 63)] {
            check_gradient(
                &head,
                &alloc::format!("candidate_w1[{i}][{j}]"),
                grads.candidate_w1[i][j],
                |h, d| h.candidate_w1[i][j] += d,
            );
        }
        // ...and each of the four scalar rows, named individually so a
        // mis-numbered row (budget read as match-class, say) fails on the
        // row it actually broke.
        for i in K..CANDIDATE_INPUT_DIM {
            // Non-vacuity per row, not per entry: an individual column's
            // gradient is legitimately zero wherever the tower's ReLU is
            // closed, but a scalar row that is dead in *every* column would
            // make its whole check pass on zeros.
            let row_max = grads.candidate_w1[i]
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(
                row_max > 0.0,
                "scalar row {i} must be live in the fixture, or this check is vacuous"
            );
            for &j in &[0usize, 29, 63] {
                check_gradient(
                    &head,
                    &alloc::format!("candidate_w1[scalar row {i}][{j}]"),
                    grads.candidate_w1[i][j],
                    |h, d| h.candidate_w1[i][j] += d,
                );
            }
        }
        for &j in &[0usize, 40, 63] {
            check_gradient(
                &head,
                &alloc::format!("candidate_b1[{j}]"),
                grads.candidate_b1[j],
                |h, d| h.candidate_b1[j] += d,
            );
        }
    }

    #[test]
    fn backward_should_scale_linearly_with_the_loss_gradient() {
        // `d_score` is dLoss/dscore and enters every term exactly once, so
        // doubling it must double every accumulated gradient. A term that
        // forgot to multiply by it (or multiplied twice) breaks here.
        let head = asymmetric_head();
        let emb = embeddings();
        let c = candidate(&head);
        let acts = Activations::forward(&head, &emb, &c);

        let mut one = HeadGradients::zero();
        let _ = head.backward(&acts, 1.0, &mut one);
        let mut two = HeadGradients::zero();
        let _ = head.backward(&acts, 2.0, &mut two);

        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                assert!(
                    (two.interaction[i][j] - 2.0 * one.interaction[i][j]).abs() < 1e-5,
                    "interaction[{i}][{j}] must be linear in d_score"
                );
            }
        }
        for j in 0..HIDDEN_DIM {
            assert!((two.trunk_b[j] - 2.0 * one.trunk_b[j]).abs() < 1e-5);
        }
    }

    #[test]
    fn accumulate_should_sum_over_samples_rather_than_overwrite() {
        let head = asymmetric_head();
        let emb = embeddings();
        let c = candidate(&head);
        let acts = Activations::forward(&head, &emb, &c);

        let mut once = HeadGradients::zero();
        let _ = head.backward(&acts, 1.0, &mut once);
        let mut twice = HeadGradients::zero();
        let _ = head.backward(&acts, 1.0, &mut twice);
        let _ = head.backward(&acts, 1.0, &mut twice);

        for k in 0..EMBED_DIM {
            assert!(
                (twice.mask_bias_proj[k] - 2.0 * once.mask_bias_proj[k]).abs() < 1e-5,
                "a second backward into the same accumulator must add, not replace"
            );
        }
    }

    #[test]
    fn trained_parameters_should_round_trip_through_the_flat_vector() {
        let head = asymmetric_head();
        let params = head.trained_parameters();
        assert_eq!(
            params.len(),
            SaturationHead::trained_parameter_count(),
            "the closed-form count must match what the visitor actually visits"
        );
        let mut restored = SaturationHead::new();
        restored.load_trained_parameters(&params);

        // Every trained tensor comes back...
        assert_eq!(restored.candidate_w1, head.candidate_w1);
        assert_eq!(restored.candidate_b1, head.candidate_b1);
        assert_eq!(restored.trunk_w, head.trunk_w);
        assert_eq!(restored.trunk_b, head.trunk_b);
        assert_eq!(restored.candidate_proj_w, head.candidate_proj_w);
        assert_eq!(restored.candidate_proj_b, head.candidate_proj_b);
        assert_eq!(restored.mask_mlp_w1, head.mask_mlp_w1);
        assert_eq!(restored.mask_mlp_b1, head.mask_mlp_b1);
        assert_eq!(restored.mask_mlp_w2, head.mask_mlp_w2);
        assert_eq!(restored.mask_mlp_b2, head.mask_mlp_b2);
        assert_eq!(restored.interaction, head.interaction);
        assert_eq!(restored.mask_bias_proj, head.mask_bias_proj);
        assert_eq!(restored.rule_proj_w, head.rule_proj_w);
        assert_eq!(restored.rule_proj_b, head.rule_proj_b);

        // ...and scoring agrees bit for bit, which is the property a
        // checkpoint boundary actually has to have.
        let emb = embeddings();
        let c = candidate(&head);
        assert_eq!(
            restored.score_candidate(&emb, &c),
            head.score_candidate(&emb, &c)
        );
    }

    #[test]
    fn every_trained_parameter_should_be_reachable_from_the_flat_vector() {
        // A tensor missing from `for_each_trained_parameter` would still
        // round-trip above (both sides skip it identically) — this catches
        // it: perturb the whole flat vector and require every trained
        // tensor to have moved.
        let head = asymmetric_head();
        let params: Vec<f32> = head.trained_parameters().iter().map(|w| w + 1.0).collect();
        let mut moved = SaturationHead::new();
        moved.load_trained_parameters(&params);
        for (name, same) in [
            ("candidate_w1", moved.candidate_w1 == head.candidate_w1),
            ("candidate_b1", moved.candidate_b1 == head.candidate_b1),
            ("trunk_w", moved.trunk_w == head.trunk_w),
            ("trunk_b", moved.trunk_b == head.trunk_b),
            (
                "candidate_proj_w",
                moved.candidate_proj_w == head.candidate_proj_w,
            ),
            (
                "candidate_proj_b",
                moved.candidate_proj_b == head.candidate_proj_b,
            ),
            ("mask_mlp_w1", moved.mask_mlp_w1 == head.mask_mlp_w1),
            ("mask_mlp_b1", moved.mask_mlp_b1 == head.mask_mlp_b1),
            ("mask_mlp_w2", moved.mask_mlp_w2 == head.mask_mlp_w2),
            ("mask_mlp_b2", moved.mask_mlp_b2 == head.mask_mlp_b2),
            ("interaction", moved.interaction == head.interaction),
            (
                "mask_bias_proj",
                moved.mask_bias_proj == head.mask_bias_proj,
            ),
            ("rule_proj_w", moved.rule_proj_w == head.rule_proj_w),
            ("rule_proj_b", moved.rule_proj_b == head.rule_proj_b),
        ] {
            assert!(!same, "{name} is not in the checkpoint's parameter order");
        }
    }

    #[test]
    fn clear_should_zero_every_accumulated_tensor() {
        let head = asymmetric_head();
        let emb = embeddings();
        let c = candidate(&head);
        let acts = Activations::forward(&head, &emb, &c);
        let mut grads = HeadGradients::zero();
        let d_rule = head.backward(&acts, 1.0, &mut grads);
        let (l, r) = sides();
        grads.accumulate_rule_proj(&rule_concat(&l, &r), &d_rule);
        grads.clear();
        let fresh = HeadGradients::zero();
        assert_eq!(grads.interaction, fresh.interaction);
        assert_eq!(grads.rule_proj_w, fresh.rule_proj_w);
        assert_eq!(grads.candidate_w1, fresh.candidate_w1);
        assert_eq!(grads.trunk_w, fresh.trunk_w);
        assert_eq!(grads.mask_mlp_w1, fresh.mask_mlp_w1);
        assert_eq!(grads.mask_mlp_w2, fresh.mask_mlp_w2);
        assert_eq!(grads.candidate_proj_w, fresh.candidate_proj_w);
        assert_eq!(grads.mask_bias_proj, fresh.mask_bias_proj);
        assert_eq!(grads.rule_proj_b, fresh.rule_proj_b);
    }

    #[test]
    fn apply_sgd_should_move_the_score_downhill_for_a_positive_loss_gradient() {
        // The end-to-end statement the finite-difference checks imply but do
        // not state: one step against a `d_score = +1` gradient lowers the
        // score at the same input.
        let mut head = asymmetric_head();
        let emb = embeddings();
        let c = candidate(&head);
        let before = score_of(&head, &emb, &c);

        let acts = Activations::forward(&head, &emb, &c);
        let mut grads = HeadGradients::zero();
        let d_rule = head.backward(&acts, 1.0, &mut grads);
        let (l, r) = sides();
        grads.accumulate_rule_proj(&rule_concat(&l, &r), &d_rule);
        head.apply_sgd(&grads, 1e-3, 0.0);

        let after = score_of(&head, &emb, &c);
        assert!(
            after < before,
            "one SGD step against a positive dLoss/dscore must lower the score: \
             {before} -> {after}"
        );
    }

    #[test]
    fn apply_sgd_should_decay_every_weight_not_only_the_ones_a_sample_touched() {
        // Mirrors `guide_linear`'s test of the same name: with an all-zero
        // gradient, an L2 step must still shrink every weight.
        let mut head = asymmetric_head();
        let before = head.clone();
        let grads = HeadGradients::zero();
        head.apply_sgd(&grads, 0.1, 0.5);

        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                assert!(
                    head.interaction[i][j].abs() < before.interaction[i][j].abs(),
                    "interaction[{i}][{j}] must decay with zero gradient"
                );
            }
        }
        for i in 0..RULE_CONCAT_DIM {
            for k in 0..EMBED_DIM {
                assert!(head.rule_proj_w[i][k].abs() < before.rule_proj_w[i][k].abs());
            }
        }
        // Biases are not decayed.
        assert_eq!(head.candidate_b1, before.candidate_b1);
        assert_eq!(head.rule_proj_b, before.rule_proj_b);
    }
}
