//! Optimization as an endomorphism on the IR.
//!
//! An optimizer takes a term and returns a term denoting the same function.
//! That is the whole contract, and stating it as a type is what makes two
//! things expressible that were not:
//!
//! - **Not optimizing is a value.** `kernel_raw!` means "the identity
//!   optimizer", and until there was a type for optimizers it could only be a
//!   branch that declined to call a function. [`Identity`] is that value.
//! - **Pipelines compose.** The runtime tier is `lower Dwrt`, then `unroll
//!   Reduce`, then `saturate` — three steps written out by hand at one call
//!   site, in an order a comment explains and nothing enforces. As
//!   [`Optimize`] values they are [`Then`], and the order is the expression.
//!
//! [`Optimize`] and [`Then`] form a monoid: `Then(Identity, p)` and
//! `Then(p, Identity)` both do what `p` does, and nesting `Then` either way
//! associates. That is not decoration — it is why an optimizer-shaped hole
//! can always be filled, including with "do nothing", which is the arm a
//! measurement needs when asking what optimization is worth.
//!
//! Note what is deliberately absent: nothing here mentions e-graphs, cost
//! models, budgets or rule sets. Those belong to one particular optimizer
//! (`pixelflow_search`'s), not to the notion.

use crate::dag::Rooted;
use crate::expr::{Environment, ExprData, Term};

/// What one [`Optimize`] step did.
///
/// Three outcomes rather than `Option`, because "I had nothing to do" and "I
/// cannot handle this term" differ in exactly one way that matters: whether
/// *later* steps in a pipeline should still run. Collapsing them silently
/// changes what a pipeline produces when a middle step bails.
#[derive(Clone)]
pub enum Rewritten {
    /// A new term, denoting what the input denoted, together with the
    /// declarations its leaves index. The environment travels with it because
    /// an optimizer may renumber slots — an extraction rebuilds leaves in
    /// whatever order it visits them — and a graph paired with the wrong
    /// table reads plausible, wrong memory.
    Changed(Rooted<ExprData>, Environment),
    /// Nothing to do here; the input stands and the pipeline continues.
    ///
    /// Distinct from returning a clone of the input: an optimizer that has no
    /// work should not cost a copy of the whole graph to say so.
    Unchanged,
    /// This term is outside what this optimizer models, and no later step
    /// should run either. The caller keeps its original input.
    ///
    /// Always available as an answer, because optimization is never required
    /// for correctness — a declined term compiles unoptimized.
    Declined,
}

// A rewritten graph is not `Debug` (it is large, and printing one is never
// what a caller wants), so report the outcome and the size rather than the
// term.
impl core::fmt::Debug for Rewritten {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Changed(rooted, _) => {
                write!(f, "Changed({} nodes)", rooted.len())
            }
            Self::Unchanged => f.write_str("Unchanged"),
            Self::Declined => f.write_str("Declined"),
        }
    }
}

impl Rewritten {
    /// The optimized term, or `None` when the input stands unchanged.
    ///
    /// Callers that hold the input already want this: `None` costs them
    /// nothing, where a `Changed` clone of the input would.
    #[must_use]
    pub fn into_changed(self) -> Option<(Rooted<ExprData>, Environment)> {
        match self {
            Self::Changed(rooted, env) => Some((rooted, env)),
            Self::Unchanged | Self::Declined => None,
        }
    }

    /// Whether a pipeline should keep going after this outcome.
    #[must_use]
    pub fn continues(&self) -> bool {
        !matches!(self, Self::Declined)
    }
}

/// A denotation-preserving endomorphism on the IR.
///
/// Implementors owe one property, and it is not checkable by this trait:
/// the returned term must denote the same function as the input. Everything
/// else — how hard it tries, what it costs, whether it does anything at all —
/// is the implementor's business.
pub trait Optimize {
    /// Rewrite the term reachable from `term`'s root.
    fn optimize(&mut self, term: Term<'_>) -> Rewritten;
}

/// The optimizer that does nothing — `kernel_raw!` as a value.
///
/// The unit of [`Then`], and the control arm of any measurement asking what
/// an optimizer is worth: the same backend, the same term, nothing rewritten.
#[derive(Clone, Copy, Debug, Default)]
pub struct Identity;

impl Optimize for Identity {
    fn optimize(&mut self, _term: Term<'_>) -> Rewritten {
        Rewritten::Unchanged
    }
}

/// Run `A`, then `B` on whatever `A` produced.
///
/// `B` sees `A`'s output when `A` changed the term and the original when it
/// did not; if `A` declines, `B` does not run and the whole composition
/// declines. That short-circuit is the reason [`Rewritten`] distinguishes
/// declining from doing nothing: a pipeline whose first step cannot lower a
/// construct must not hand the un-lowered term to the steps that assume it is
/// gone.
#[derive(Clone, Copy, Debug, Default)]
pub struct Then<A, B>(pub A, pub B);

impl<A: Optimize, B: Optimize> Optimize for Then<A, B> {
    fn optimize(&mut self, term: Term<'_>) -> Rewritten {
        match self.0.optimize(term) {
            Rewritten::Declined => Rewritten::Declined,
            Rewritten::Unchanged => self.1.optimize(term),
            Rewritten::Changed(rooted, env) => {
                match self.1.optimize(Term::new(rooted.entry(), &env)) {
                    // `B` declining does not discard `A`'s work: `A` already
                    // produced a valid term, and the caller's input is not
                    // more correct than it — only less optimized.
                    Rewritten::Declined | Rewritten::Unchanged => Rewritten::Changed(rooted, env),
                    changed => changed,
                }
            }
        }
    }
}

/// Sequence optimizers left to right: `pipeline![a, b, c]` is
/// `Then(a, Then(b, c))`.
#[macro_export]
macro_rules! pipeline {
    ($single:expr $(,)?) => { $single };
    ($first:expr, $($rest:expr),+ $(,)?) => {
        $crate::optimize::Then($first, $crate::pipeline![$($rest),+])
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OpKind;
    use crate::expr::ExprBuilder;

    /// Counts calls, and optionally rewrites `x` to `x + 1.0`.
    struct Bump {
        calls: usize,
        outcome: fn(Term<'_>) -> Rewritten,
    }

    fn bump(term: Term<'_>) -> Rewritten {
        let mut out = ExprBuilder::new();
        let copied = out.splice(term);
        let one = out.push_const(1.0);
        let root = out.push_binary(OpKind::Add, copied, one);
        let (rooted, env) = out.finish(&[root]);
        Rewritten::Changed(rooted, env)
    }
    fn nothing(_: Term<'_>) -> Rewritten {
        Rewritten::Unchanged
    }
    fn decline(_: Term<'_>) -> Rewritten {
        Rewritten::Declined
    }

    impl Optimize for Bump {
        fn optimize(&mut self, term: Term<'_>) -> Rewritten {
            self.calls += 1;
            (self.outcome)(term)
        }
    }

    fn bumper(outcome: fn(Term<'_>) -> Rewritten) -> Bump {
        Bump { calls: 0, outcome }
    }

    fn seed() -> (Rooted<ExprData>, Environment) {
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        b.finish(&[x])
    }

    #[test]
    fn identity_is_the_unit_of_then() {
        let (rooted, env) = seed();
        let term = Term::new(rooted.entry(), &env);
        let mut left = Then(Identity, bumper(bump));
        let mut right = Then(bumper(bump), Identity);
        let l = left.optimize(term).into_changed().expect("changed");
        let r = right.optimize(term).into_changed().expect("changed");
        assert_eq!(
            l.0.len(),
            r.0.len(),
            "Identity on either side must not change the result"
        );
    }

    #[test]
    fn identity_alone_leaves_the_term_alone() {
        let (rooted, env) = seed();
        let term = Term::new(rooted.entry(), &env);
        assert!(Identity.optimize(term).into_changed().is_none());
    }

    /// The short-circuit: a declining step must stop the ones after it, or a
    /// pipeline hands a term to a step whose precondition the declining step
    /// was responsible for establishing.
    #[test]
    fn declining_stops_later_steps() {
        let (rooted, env) = seed();
        let term = Term::new(rooted.entry(), &env);
        let mut pipe = Then(bumper(decline), bumper(bump));
        assert!(matches!(pipe.optimize(term), Rewritten::Declined));
        assert_eq!(pipe.0.calls, 1);
        assert_eq!(pipe.1.calls, 0, "the step after a decline must not run");
    }

    /// Doing nothing is not declining: later steps still run.
    #[test]
    fn unchanged_does_not_stop_later_steps() {
        let (rooted, env) = seed();
        let term = Term::new(rooted.entry(), &env);
        let mut pipe = Then(bumper(nothing), bumper(bump));
        assert!(pipe.optimize(term).into_changed().is_some());
        assert_eq!(pipe.1.calls, 1, "a no-op step must not stop the pipeline");
    }

    /// A later step declining keeps the earlier step's work rather than
    /// throwing the pipeline back to the caller's input.
    #[test]
    fn a_later_decline_keeps_earlier_work() {
        let (rooted, env) = seed();
        let term = Term::new(rooted.entry(), &env);
        let mut pipe = Then(bumper(bump), bumper(decline));
        let (out, _) = pipe.optimize(term).into_changed().expect("kept");
        assert!(
            out.len() > rooted.len(),
            "the first step's rewrite must survive"
        );
    }

    #[test]
    fn pipeline_macro_associates_left_to_right() {
        let (rooted, env) = seed();
        let term = Term::new(rooted.entry(), &env);
        let mut pipe = crate::pipeline![bumper(bump), bumper(bump), bumper(bump)];
        let (out, _) = pipe.optimize(term).into_changed().expect("changed");
        // Each bump adds a Const and an Add to whatever it was handed.
        assert_eq!(out.len(), rooted.len() + 6);
    }
}
