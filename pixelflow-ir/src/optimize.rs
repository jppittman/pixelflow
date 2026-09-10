//! Optimization as an endomorphism on an owned expression graph.

use crate::expr::ExprGraph;

/// What one [`Optimize`] step did.
#[derive(Clone)]
pub enum Rewritten {
    /// A new graph, denoting what the input denoted.
    Changed(ExprGraph),
    /// Nothing to do here; the input stands and the pipeline continues.
    Unchanged,
    /// This term is outside what this optimizer models; later steps stop.
    Declined,
}

impl core::fmt::Debug for Rewritten {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Changed(graph) => write!(f, "Changed({} nodes)", graph.dag().len()),
            Self::Unchanged => f.write_str("Unchanged"),
            Self::Declined => f.write_str("Declined"),
        }
    }
}

impl Rewritten {
    #[must_use]
    pub fn into_changed(self) -> Option<ExprGraph> {
        match self {
            Self::Changed(graph) => Some(graph),
            Self::Unchanged | Self::Declined => None,
        }
    }

    #[must_use]
    pub fn continues(&self) -> bool {
        !matches!(self, Self::Declined)
    }
}

/// A denotation-preserving endomorphism on an expression graph.
pub trait Optimize {
    fn optimize(&mut self, graph: &ExprGraph) -> Rewritten;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Identity;

impl Optimize for Identity {
    fn optimize(&mut self, _graph: &ExprGraph) -> Rewritten {
        Rewritten::Unchanged
    }
}

/// Run `A`, then `B` on whatever `A` produced.
#[derive(Clone, Copy, Debug, Default)]
pub struct Then<A, B>(pub A, pub B);

impl<A: Optimize, B: Optimize> Optimize for Then<A, B> {
    fn optimize(&mut self, graph: &ExprGraph) -> Rewritten {
        match self.0.optimize(graph) {
            Rewritten::Declined => Rewritten::Declined,
            Rewritten::Unchanged => self.1.optimize(graph),
            Rewritten::Changed(first) => match self.1.optimize(&first) {
                Rewritten::Declined | Rewritten::Unchanged => Rewritten::Changed(first),
                changed => changed,
            },
        }
    }
}

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
    use crate::ExprBuilder;
    use crate::dag::Builder;
    use crate::expr::{ExprBuilderExt, copy_subgraph};
    use crate::kind::OpKind;

    struct Bump {
        calls: usize,
        outcome: fn(&ExprGraph) -> Rewritten,
    }

    fn bump(graph: &ExprGraph) -> Rewritten {
        let mut builder = Builder::new();
        let root = copy_subgraph(&mut builder, graph.root());
        let one = builder.push_const(1.0);
        let root = builder.push_binary(OpKind::Add, root, one);
        Rewritten::Changed(ExprGraph::new(
            builder.finish(&[root]),
            graph.environment().clone(),
        ))
    }
    fn nothing(_: &ExprGraph) -> Rewritten {
        Rewritten::Unchanged
    }
    fn decline(_: &ExprGraph) -> Rewritten {
        Rewritten::Declined
    }

    impl Optimize for Bump {
        fn optimize(&mut self, graph: &ExprGraph) -> Rewritten {
            self.calls += 1;
            (self.outcome)(graph)
        }
    }

    fn seed() -> ExprGraph {
        let mut builder = ExprBuilder::new();
        let x = builder.var(0);
        builder.finish_one(x)
    }

    #[test]
    fn identity_is_the_unit_of_then() {
        let graph = seed();
        let mut left = Then(
            Identity,
            Bump {
                calls: 0,
                outcome: bump,
            },
        );
        let mut right = Then(
            Bump {
                calls: 0,
                outcome: bump,
            },
            Identity,
        );
        let l = left.optimize(&graph).into_changed().expect("changed");
        let r = right.optimize(&graph).into_changed().expect("changed");
        assert_eq!(l.dag().len(), r.dag().len());
        assert_eq!(*l.root(), *r.root());
    }

    #[test]
    fn identity_alone_leaves_the_term_alone() {
        let graph = seed();
        assert!(Identity.optimize(&graph).into_changed().is_none());
    }

    #[test]
    fn declining_stops_later_steps() {
        let graph = seed();
        let mut pipe = Then(
            Bump {
                calls: 0,
                outcome: decline,
            },
            Bump {
                calls: 0,
                outcome: bump,
            },
        );
        assert!(matches!(pipe.optimize(&graph), Rewritten::Declined));
        assert_eq!(pipe.0.calls, 1);
        assert_eq!(pipe.1.calls, 0);
    }

    #[test]
    fn unchanged_does_not_stop_later_steps() {
        let graph = seed();
        let mut pipe = Then(
            Bump {
                calls: 0,
                outcome: nothing,
            },
            Bump {
                calls: 0,
                outcome: bump,
            },
        );
        assert!(pipe.optimize(&graph).into_changed().is_some());
        assert_eq!(pipe.1.calls, 1);
    }

    #[test]
    fn a_later_decline_keeps_earlier_work() {
        let graph = seed();
        let mut pipe = Then(
            Bump {
                calls: 0,
                outcome: bump,
            },
            Bump {
                calls: 0,
                outcome: decline,
            },
        );
        let out = pipe.optimize(&graph).into_changed().expect("kept");
        assert!(out.dag().len() > graph.dag().len());
    }

    #[test]
    fn pipeline_macro_associates_left_to_right() {
        let graph = seed();
        let mut pipe = crate::pipeline![
            Bump {
                calls: 0,
                outcome: bump
            },
            Bump {
                calls: 0,
                outcome: bump
            },
            Bump {
                calls: 0,
                outcome: bump
            },
        ];
        let out = pipe.optimize(&graph).into_changed().expect("changed");
        assert_eq!(out.dag().len(), graph.dag().len() + 6);
    }
}
