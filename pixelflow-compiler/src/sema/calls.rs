//! The call graph over the block's `fn`s, which is a DAG.

use crate::ast::{BlockExpr, Expr, FnItem, Name, Stmt};
use proc_macro2::Span;
use std::collections::HashMap;

/// The call graph over the block's `fn`s is a DAG: a helper is inlined at
/// each call, and a cycle would inline forever.
pub(super) fn refuse_recursion(fns: &[FnItem]) -> syn::Result<()> {
    let names: Vec<Name> = fns.iter().map(|f| Name::of(&f.name)).collect();
    let calls: HashMap<Name, Vec<(Name, Span)>> = fns
        .iter()
        .map(|f| {
            let mut out = Vec::new();
            collect_calls(&f.body, &names, &mut out);
            (Name::of(&f.name), out)
        })
        .collect();
    let mut walk = CallGraphWalk {
        calls: &calls,
        finished: Vec::new(),
        path: Vec::new(),
    };
    for name in &names {
        walk.visit(name)?;
    }
    Ok(())
}

/// A depth-first walk of the call graph, refusing the first edge that
/// closes a cycle.
struct CallGraphWalk<'a> {
    calls: &'a HashMap<Name, Vec<(Name, Span)>>,
    /// Every `fn` whose reachable calls are known to be acyclic.
    finished: Vec<Name>,
    /// The `fn`s whose bodies the walk is inside, outermost first.
    path: Vec<Name>,
}

impl CallGraphWalk<'_> {
    fn visit(&mut self, name: &Name) -> syn::Result<()> {
        if self.finished.contains(name) {
            return Ok(());
        }
        self.path.push(name.clone());
        for (callee, span) in &self.calls[name] {
            if let Some(start) = self.path.iter().position(|f| f == callee) {
                let cycle = self.path[start..]
                    .iter()
                    .map(Name::to_string)
                    .collect::<Vec<_>>()
                    .join("` calls `");
                return Err(syn::Error::new(
                    *span,
                    format!(
                        "recursion is refused: the language is a DAG\n\
                         \n\
                         note: `{cycle}` calls `{callee}`\n\
                         note: a helper is inlined at each call, so a cycle would inline \
                         forever; a bounded reduction is a fold, `(a..b).map(|i| e).sum()`"
                    ),
                ));
            }
            self.visit(callee)?;
        }
        self.path.pop();
        self.finished.push(name.clone());
        Ok(())
    }
}

/// Every call in `expr` to one of the block's `fn`s, with its span.
fn collect_calls(expr: &Expr, fns: &[Name], out: &mut Vec<(Name, Span)>) {
    match expr {
        Expr::Ident(_) | Expr::Literal(_) => {}
        Expr::Binary(binary) => {
            collect_calls(&binary.lhs, fns, out);
            collect_calls(&binary.rhs, fns, out);
        }
        Expr::Unary(unary) => collect_calls(&unary.operand, fns, out),
        Expr::MethodCall(call) => {
            collect_calls(&call.receiver, fns, out);
            for arg in &call.args {
                collect_calls(arg, fns, out);
            }
        }
        Expr::Project(project) => collect_calls(&project.arg, fns, out),
        Expr::Call(call) => {
            let name = Name::of(&call.func);
            if fns.contains(&name) {
                out.push((name, call.func.span()));
            }
            for arg in &call.args {
                collect_calls(arg, fns, out);
            }
        }
        Expr::If(choice) => {
            collect_calls(&choice.cond, fns, out);
            collect_block_calls(&choice.then_branch, fns, out);
            collect_calls(&choice.else_branch, fns, out);
        }
        // A bound is constant, so it calls nothing; a body may.
        Expr::Fold(fold) => collect_calls(&fold.body, fns, out),
        Expr::Cast(cast) => collect_calls(&cast.operand, fns, out),
        Expr::Field(field) => collect_calls(&field.base, fns, out),
        Expr::Block(block) => collect_block_calls(block, fns, out),
        Expr::Paren(inner) => collect_calls(inner, fns, out),
    }
}

fn collect_block_calls(block: &BlockExpr, fns: &[Name], out: &mut Vec<(Name, Span)>) {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Let(lets) => {
                for let_stmt in lets {
                    collect_calls(&let_stmt.init, fns, out);
                }
            }
            Stmt::Expr(expr) => collect_calls(expr, fns, out),
        }
    }
    if let Some(expr) = &block.expr {
        collect_calls(expr, fns, out);
    }
}
