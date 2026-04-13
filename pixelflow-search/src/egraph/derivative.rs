//! Symbolic Differentiation Rules
//!
//! These rules define how to apply the chain rule and other differentiation
//! identities directly inside the E-Graph.

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::rewrite::{Rewrite, RewriteAction};
use pixelflow_ir::kind::OpKind;

// Note: To implement this, we need a new OpKind representing the "Derivative" operator
// D_wrt(expr, var_idx). When the E-Graph sees a D_wrt node, it applies these rules to
// expand it into basic arithmetic, and the constant folding rules will clean up the zeros.
