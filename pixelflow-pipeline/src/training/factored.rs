//! # Expression parsing and serialization utilities for NNUE training.
//!
//! Provides parsers for two expression syntaxes:
//! - **S-expression**: `Add(Mul(Var(0), Var(1)), Var(2))` (test-only repro format)
//! - **Kernel code**: `(X * Y) + Z` (human-readable, round-trips with
//!   `parse_kernel_code`/`graph_to_kernel_code`)

use pixelflow_ir::{EmitStyle, ExprBuilder, ExprData, ExprGraph, ExprHandle, Node, OpKind};

// ============================================================================
// Expression Parsing (for loading training data)
// ============================================================================

/// Parse an expression from a string representation.
///
/// Test-only helper for older logged repros in `OpName(child1, child2, ...)`
/// form such as `Add(Mul(Var(0), Var(1)), Var(2))`.
#[cfg(test)]
pub fn parse_expr(s: &str) -> Option<ExprGraph> {
    let mut builder = ExprBuilder::new();
    let root = parse_expr_into(s, &mut builder)?;
    Some(builder.finish_one(root))
}

/// Recursive S-expression parser that builds directly into an [`ExprBuilder`].
#[cfg(test)]
fn parse_expr_into(s: &str, builder: &mut ExprBuilder) -> Option<ExprHandle> {
    let s = s.trim();

    if let Some(inner) = s.strip_prefix("Var(").and_then(|r| r.strip_suffix(')')) {
        let idx: u8 = inner.trim().parse().ok()?;
        return Some(builder.var(idx));
    }
    if let Some(inner) = s.strip_prefix("Const(").and_then(|r| r.strip_suffix(')')) {
        let val: f32 = inner.trim().parse().ok()?;
        return Some(builder.constant(val));
    }

    let paren_pos = s.find('(')?;
    let op_name = s[..paren_pos].to_lowercase();
    let inner = &s[paren_pos + 1..s.len() - 1];
    let children = split_args(inner);

    // `fract` and `hypot` are compound ops (no hardware instruction backs
    // them, see pixelflow-ir/src/kernel.rs), so they no longer have a single
    // OpKind to construct via unary/binary builder methods below. Build the same
    // primitive subgraph pixelflow_ir::backend::compounds::Compounds does.
    match op_name.as_str() {
        "fract" if children.len() == 1 => {
            // fract(x) = x - floor(x)
            let a = parse_expr_into(children[0], builder)?;
            let floor_a = builder.unary(OpKind::Floor, a);
            return Some(builder.binary(OpKind::Sub, a, floor_a));
        }
        "hypot" if children.len() == 2 => {
            // hypot(x, y) = sqrt(x*x + y*y)
            let a = parse_expr_into(children[0], builder)?;
            let b = parse_expr_into(children[1], builder)?;
            let aa = builder.binary(OpKind::Mul, a, a);
            let bb = builder.binary(OpKind::Mul, b, b);
            let sum = builder.binary(OpKind::Add, aa, bb);
            return Some(builder.unary(OpKind::Sqrt, sum));
        }
        _ => {}
    }

    let op = parse_op_kind(&op_name)?;
    match (op.arity(), children.len()) {
        (1, 1) => {
            let a = parse_expr_into(children[0], builder)?;
            Some(builder.unary(op, a))
        }
        (2, 2) => {
            let a = parse_expr_into(children[0], builder)?;
            let b = parse_expr_into(children[1], builder)?;
            Some(builder.binary(op, a, b))
        }
        (3, 3) => {
            let a = parse_expr_into(children[0], builder)?;
            let b = parse_expr_into(children[1], builder)?;
            let c = parse_expr_into(children[2], builder)?;
            Some(builder.ternary(op, a, b, c))
        }
        _ => None,
    }
}

/// Parse operation name to OpKind.
#[cfg(test)]
fn parse_op_kind(name: &str) -> Option<OpKind> {
    match name.to_lowercase().as_str() {
        "add" => Some(OpKind::Add),
        "sub" => Some(OpKind::Sub),
        "mul" => Some(OpKind::Mul),
        "div" => Some(OpKind::Div),
        "neg" => Some(OpKind::Neg),
        "sqrt" => Some(OpKind::Sqrt),
        "rsqrt" => Some(OpKind::Rsqrt),
        "abs" => Some(OpKind::Abs),
        "min" => Some(OpKind::Min),
        "max" => Some(OpKind::Max),
        "muladd" | "mul_add" | "fma" => Some(OpKind::MulAdd),
        "recip" => Some(OpKind::Recip),
        "floor" => Some(OpKind::Floor),
        "ceil" => Some(OpKind::Ceil),
        "round" => Some(OpKind::Round),
        "sin" => Some(OpKind::Sin),
        "cos" => Some(OpKind::Cos),
        "tan" => Some(OpKind::Tan),
        "asin" => Some(OpKind::Asin),
        "acos" => Some(OpKind::Acos),
        "atan" => Some(OpKind::Atan),
        "atan2" => Some(OpKind::Atan2),
        "exp" => Some(OpKind::Exp),
        "exp2" => Some(OpKind::Exp2),
        "ln" => Some(OpKind::Ln),
        "log2" => Some(OpKind::Log2),
        "log10" => Some(OpKind::Log10),
        "pow" => Some(OpKind::Pow),
        _ => None,
    }
}

/// Split comma-separated arguments, respecting nested parentheses.
#[cfg(test)]
fn split_args(s: &str) -> Vec<&str> {
    let mut args = Vec::new();
    let mut depth = 0;
    let mut start = 0;

    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                args.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }

    if start < s.len() {
        args.push(s[start..].trim());
    }

    args
}

// ============================================================================
// Kernel Code Parsing
// ============================================================================
//
// Grammar:
//   expr     ::= additive
//   additive ::= multiplicative (('+' | '-') multiplicative)*
//   mult     ::= postfix (('*' | '/') postfix)*
//   postfix  ::= primary ('.' method)*
//   method   ::= IDENT '(' expr? ')'
//   primary  ::= '(' expr ')' | '-' postfix | VAR | NUM
//   VAR      ::= 'X' | 'Y' | 'Z' | 'W'
//   NUM      ::= float literal

struct GraphInterner {
    builder: ExprBuilder,
}

impl GraphInterner {
    fn new() -> Self {
        Self {
            builder: ExprBuilder::new(),
        }
    }

    fn var(&mut self, index: u8) -> ExprHandle {
        self.builder.var(index)
    }

    fn constant(&mut self, value: f32) -> ExprHandle {
        self.builder.constant(value)
    }

    fn unary(&mut self, op: OpKind, a: ExprHandle) -> ExprHandle {
        self.builder.unary(op, a)
    }

    fn binary(&mut self, op: OpKind, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.builder.binary(op, a, b)
    }

    fn ternary(&mut self, op: OpKind, a: ExprHandle, b: ExprHandle, c: ExprHandle) -> ExprHandle {
        self.builder.ternary(op, a, b, c)
    }

    fn finish(self, root: ExprHandle) -> ExprGraph {
        self.builder.finish_one(root)
    }
}

enum ParseOp {
    PrefixNeg,
    Binary(OpKind),
    GroupParen,
    GroupMethod { op: OpKind, commas: usize },
}

struct GraphKernelParser<'a> {
    input: &'a str,
    pos: usize,
    interner: GraphInterner,
}

impl<'a> GraphKernelParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            pos: 0,
            interner: GraphInterner::new(),
        }
    }

    fn parse(mut self) -> Option<ExprGraph> {
        let root = self.parse_expr()?;
        self.skip_ws();
        (self.pos == self.input.len()).then(|| self.interner.finish(root))
    }

    fn parse_expr(&mut self) -> Option<ExprHandle> {
        let mut values = Vec::new();
        let mut ops = Vec::new();
        let mut expecting_operand = true;

        loop {
            self.skip_ws();

            if expecting_operand {
                match self.peek_char()? {
                    '(' => {
                        self.pos += 1;
                        ops.push(ParseOp::GroupParen);
                    }
                    '-' => {
                        self.pos += 1;
                        ops.push(ParseOp::PrefixNeg);
                    }
                    'X' | 'Y' | 'Z' | 'W' => {
                        let var = self.parse_variable_id()?;
                        values.push(self.interner.var(var));
                        expecting_operand = false;
                    }
                    c if c.is_ascii_digit() || c == '.' => {
                        let number = self.parse_number_literal()?;
                        values.push(self.interner.constant(number));
                        expecting_operand = false;
                    }
                    _ => return None,
                }
                continue;
            }

            if self.consume_char('.') {
                let method = self.parse_ident_at_pos()?;
                self.skip_ws();
                if !self.consume_char('(') {
                    return None;
                }

                let op = OpKind::from_name(method)?;
                if op.arity() == 1 {
                    self.skip_ws();
                    if !self.consume_char(')') {
                        return None;
                    }
                    let base = values.pop()?;
                    values.push(self.interner.unary(op, base));
                } else {
                    ops.push(ParseOp::GroupMethod { op, commas: 0 });
                    expecting_operand = true;
                }
                continue;
            }

            Self::reduce_prefix_negs(&mut self.interner, &mut values, &mut ops)?;
            self.skip_ws();

            match self.peek_char() {
                Some('+') | Some('-') | Some('*') | Some('/') => {
                    let op = self.parse_binary_op()?;
                    Self::reduce_binary_ops(&mut self.interner, &mut values, &mut ops, op)?;
                    ops.push(ParseOp::Binary(op));
                    expecting_operand = true;
                }
                Some(',') => {
                    self.pos += 1;
                    Self::reduce_until_group(&mut self.interner, &mut values, &mut ops)?;
                    match ops.last_mut() {
                        Some(ParseOp::GroupMethod { commas, .. }) => {
                            *commas += 1;
                            expecting_operand = true;
                        }
                        _ => return None,
                    }
                }
                Some(')') => {
                    self.pos += 1;
                    Self::reduce_until_group(&mut self.interner, &mut values, &mut ops)?;
                    match ops.pop()? {
                        ParseOp::GroupParen => {}
                        ParseOp::GroupMethod { op, commas } => {
                            let explicit_args = commas + 1;
                            if explicit_args + 1 != op.arity() {
                                return None;
                            }
                            let value = match op.arity() {
                                2 => {
                                    let b = values.pop()?;
                                    let a = values.pop()?;
                                    self.interner.binary(op, a, b)
                                }
                                3 => {
                                    let c = values.pop()?;
                                    let b = values.pop()?;
                                    let a = values.pop()?;
                                    self.interner.ternary(op, a, b, c)
                                }
                                _ => return None,
                            };
                            values.push(value);
                        }
                        ParseOp::PrefixNeg | ParseOp::Binary(_) => return None,
                    }
                }
                None => {
                    Self::reduce_until_group(&mut self.interner, &mut values, &mut ops)?;
                    if !ops.is_empty() {
                        return None;
                    }
                    return (values.len() == 1).then(|| values.pop()).flatten();
                }
                _ => return None,
            }
        }
    }

    fn reduce_prefix_negs(
        interner: &mut GraphInterner,
        values: &mut Vec<ExprHandle>,
        ops: &mut Vec<ParseOp>,
    ) -> Option<()> {
        while matches!(ops.last(), Some(ParseOp::PrefixNeg)) {
            ops.pop();
            let value = values.pop()?;
            values.push(interner.unary(OpKind::Neg, value));
        }
        Some(())
    }

    fn reduce_binary_ops(
        interner: &mut GraphInterner,
        values: &mut Vec<ExprHandle>,
        ops: &mut Vec<ParseOp>,
        incoming: OpKind,
    ) -> Option<()> {
        loop {
            match ops.last() {
                Some(ParseOp::Binary(current))
                    if Self::precedence(*current) >= Self::precedence(incoming) =>
                {
                    let current = match ops.pop()? {
                        ParseOp::Binary(op) => op,
                        _ => unreachable!(),
                    };
                    let b = values.pop()?;
                    let a = values.pop()?;
                    values.push(interner.binary(current, a, b));
                }
                Some(ParseOp::PrefixNeg) => {
                    Self::reduce_prefix_negs(interner, values, ops)?;
                }
                _ => break,
            }
        }
        Some(())
    }

    fn reduce_until_group(
        interner: &mut GraphInterner,
        values: &mut Vec<ExprHandle>,
        ops: &mut Vec<ParseOp>,
    ) -> Option<()> {
        loop {
            match ops.last() {
                Some(ParseOp::Binary(_)) => {
                    let op = match ops.pop()? {
                        ParseOp::Binary(op) => op,
                        _ => unreachable!(),
                    };
                    let b = values.pop()?;
                    let a = values.pop()?;
                    values.push(interner.binary(op, a, b));
                }
                Some(ParseOp::PrefixNeg) => {
                    Self::reduce_prefix_negs(interner, values, ops)?;
                }
                _ => break,
            }
        }
        Some(())
    }

    fn precedence(op: OpKind) -> u8 {
        match op {
            OpKind::Add | OpKind::Sub => 1,
            OpKind::Mul | OpKind::Div => 2,
            _ => 0,
        }
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek_char(), Some(c) if c.is_whitespace()) {
            self.pos += 1;
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn consume_char(&mut self, expected: char) -> bool {
        if self.peek_char() == Some(expected) {
            self.pos += expected.len_utf8();
            true
        } else {
            false
        }
    }

    fn parse_variable_id(&mut self) -> Option<u8> {
        let var = match self.peek_char()? {
            'X' => 0,
            'Y' => 1,
            'Z' => 2,
            'W' => 3,
            _ => return None,
        };
        self.pos += 1;
        Some(var)
    }

    fn parse_number_literal(&mut self) -> Option<f32> {
        let start = self.pos;
        let mut seen_digit = false;
        let mut seen_dot = false;
        let mut seen_exp = false;

        while let Some(c) = self.peek_char() {
            match c {
                '0'..='9' => {
                    seen_digit = true;
                    self.pos += 1;
                }
                '.' if !seen_dot && !seen_exp => {
                    seen_dot = true;
                    self.pos += 1;
                }
                'e' | 'E' if seen_digit && !seen_exp => {
                    seen_exp = true;
                    self.pos += 1;
                    if matches!(self.peek_char(), Some('+') | Some('-')) {
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }

        (self.pos > start && seen_digit)
            .then(|| self.input[start..self.pos].parse().ok())
            .flatten()
    }

    fn parse_ident_at_pos(&mut self) -> Option<&'a str> {
        let start = self.pos;
        let (ident, _) = parse_ident(&self.input[start..])?;
        self.pos += ident.len();
        Some(ident)
    }

    fn parse_binary_op(&mut self) -> Option<OpKind> {
        let op = match self.peek_char()? {
            '+' => OpKind::Add,
            '-' => OpKind::Sub,
            '*' => OpKind::Mul,
            '/' => OpKind::Div,
            _ => return None,
        };
        self.pos += 1;
        Some(op)
    }
}

/// Parse kernel code directly into an immutable [`ExprGraph`] with structural
/// sharing. The builder owns interning and the resulting graph owns its DAG
/// storage, so callers never carry a node index beside the expression.
///
/// Returns `None` if the input fails to parse.
pub fn parse_kernel_code(s: &str) -> Option<ExprGraph> {
    GraphKernelParser::new(s.trim()).parse()
}

/// Parse an identifier (method name). Accepts `[a-zA-Z_][a-zA-Z0-9_]*`
/// to handle op names with digits like `atan2`, `exp2`, `log2`, `log10`.
fn parse_ident(input: &str) -> Option<(&str, &str)> {
    let mut chars = input.char_indices();
    // First character must be alphabetic or underscore
    match chars.next() {
        Some((_, c)) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return None,
    }
    // Remaining characters can also include digits
    let end = chars
        .find(|(_, c)| !c.is_ascii_alphanumeric() && *c != '_')
        .map(|(i, _)| i)
        .unwrap_or(input.len());

    Some((&input[..end], &input[end..]))
}

// ============================================================================
// DAG → Kernel Code Serialization
// ============================================================================

/// Convert an expression graph into kernel code syntax.
pub fn graph_to_kernel_code(graph: &ExprGraph) -> String {
    enum Task<'a> {
        Visit(Node<'a, ExprData>),
        Emit {
            node: Node<'a, ExprData>,
            arity: usize,
        },
    }

    let mut stack = vec![Task::Visit(graph.root())];
    let mut result_stack: Vec<String> = Vec::new();

    while let Some(task) = stack.pop() {
        match task {
            Task::Visit(node) => {
                let arity = node.child_count();
                stack.push(Task::Emit { node, arity });
                let children: Vec<Node<'_, ExprData>> = node.children().collect();
                for child in children.into_iter().rev() {
                    stack.push(Task::Visit(child));
                }
            }
            Task::Emit { node, arity } => {
                let start = result_stack.len().saturating_sub(arity);
                let args: Vec<String> = result_stack.drain(start..).collect();
                let emitted = match *node {
                    ExprData::Var(0) => "X".into(),
                    ExprData::Var(1) => "Y".into(),
                    ExprData::Var(2) => "Z".into(),
                    ExprData::Var(3) => "W".into(),
                    ExprData::Var(i) => panic!(
                        "graph_to_kernel_code: variable index {} exceeds X/Y/Z/W range",
                        i
                    ),
                    ExprData::Const(bits) => format_const_kc(f32::from_bits(bits)),
                    ExprData::Param(i) => panic!(
                        "ExprData::Param({}) reached graph_to_kernel_code — substitute params first",
                        i
                    ),
                    ExprData::Buffer(b) => panic!(
                        "ExprData::Buffer({}) reached graph_to_kernel_code — memory ops require \
                         a binding table, not yet wired (M2, see KERNELS_AND_LATTICES.md)",
                        b.0
                    ),
                    ExprData::Uniform(u) => panic!(
                        "ExprData::Uniform({}) reached graph_to_kernel_code — kernel code has \
                         no block to read it from",
                        u.0
                    ),
                    ExprData::Ref(k) => panic!(
                        "ExprData::Ref({k:?}) reached graph_to_kernel_code — kernel code has \
                         no syntax for a reference; expand_refs first"
                    ),
                    ExprData::Reduce(_) => panic!(
                        "a bounded fold reached graph_to_kernel_code — kernel code has no \
                         syntax for a binder; expand_reduce first"
                    ),
                    ExprData::Op(op) if arity <= 3 => emit_op_kc(op, &args),
                    ExprData::Op(op) => panic!(
                        "graph_to_kernel_code: Nary({}) not representable in kernel code syntax",
                        op.name()
                    ),
                };
                result_stack.push(emitted);
            }
        }
    }

    result_stack
        .pop()
        .unwrap_or_else(|| panic!("graph_to_kernel_code: empty result stack"))
}

/// Emit an operation in kernel code syntax, dispatching through `emit_style()`.
fn emit_op_kc(op: OpKind, args: &[String]) -> String {
    match (op.emit_style(), args) {
        (EmitStyle::UnaryPrefix, [a]) => format!("(-{})", a),
        (EmitStyle::UnaryMethod, [a]) => format!("({}).{}()", a, op.name()),
        (EmitStyle::BinaryInfix(sym), [a, b]) => format!("({} {} {})", a, sym, b),
        (EmitStyle::BinaryMethod, [a, b]) => format!("({}).{}({})", a, op.name(), b),
        (EmitStyle::BinaryMethodNamed(method), [a, b]) => format!("({}).{}({})", a, method, b),
        (EmitStyle::TernaryMethod, [a, b, c]) => {
            format!("({}).{}({}, {})", a, op.name(), b, c)
        }
        (EmitStyle::Special, _) => panic!(
            "emit_op_kc: Special ops (Var/Const/Tuple) must be handled by caller, got {}",
            op.name()
        ),
        (style, args) => panic!(
            "emit_op_kc: arity mismatch for {}: {:?} expects different arg count, got {}",
            op.name(),
            style,
            args.len()
        ),
    }
}

/// Format a constant for kernel code syntax.
fn format_const_kc(v: f32) -> String {
    if !v.is_finite() {
        panic!("graph_to_kernel_code: non-finite constant {v} cannot be represented");
    }
    if v.is_sign_negative() && v != 0.0 {
        return format!("(-{})", format_const_kc(-v));
    }
    // Rust's Display for f32 produces the shortest round-trip representation.
    format!("{v}")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// The values the frozen corpus expressions' third and fourth variables
    /// take. Those were the Z and W coordinates; a lattice has two axes now,
    /// so the fixtures' `Var(2)`/`Var(3)` are bound as arguments — invariant
    /// across the lattice, which at a single point is the same number the
    /// coordinate carried.
    const REWRITE_BUG_ARGS: [f32; 2] = [1.3, -0.2];

    /// Replace the frozen fixtures' `Var(2)`/`Var(3)` with the values they
    /// used to carry as the Z and W coordinates.
    ///
    /// **Every** consumer of a fixture goes through this, which is the whole
    /// point. A scalar oracle used to substitute here while the JIT harness
    /// fed the emitter zeros in those lanes, so the two evaluated different
    /// programs and disagreed by three orders of magnitude — a silent numeric
    /// divergence, not a failure. `emit::compile_dag` now refuses a graph that
    /// names a retired axis, so a fixture that skipped this panics rather
    /// than diverging, which is why that pairing is no longer what defends
    /// the invariant.
    ///
    /// Constants, not uniforms: the collapse ABI is called with a null
    /// context in these harnesses, so a uniform read would fault. A constant
    /// needs no context and denotes exactly the same number either way.
    fn bind_retired_axes(graph: &ExprGraph) -> ExprGraph {
        fn copy(builder: &mut ExprBuilder, node: Node<'_, ExprData>) -> ExprHandle {
            match *node {
                ExprData::Var(index) if index >= 2 => {
                    let slot = (index - 2) as usize;
                    builder.constant(
                        *REWRITE_BUG_ARGS
                            .get(slot)
                            .unwrap_or_else(|| panic!("retired axis {index} has no fixture value")),
                    )
                }
                ExprData::Var(index) => builder.var(index),
                ExprData::Const(bits) => builder.constant(f32::from_bits(bits)),
                ExprData::Op(op) => {
                    let children: Vec<_> =
                        node.children().map(|child| copy(builder, child)).collect();
                    match children.as_slice() {
                        [a] => builder.unary(op, *a),
                        [a, b] => builder.binary(op, *a, *b),
                        [a, b, c] => builder.ternary(op, *a, *b, *c),
                        _ => panic!("retired-axis fixture contains unsupported operator arity"),
                    }
                }
                other => panic!("unexpected fixture node during retired-axis binding: {other:?}"),
            }
        }

        let mut builder = ExprBuilder::new();
        let root = copy(&mut builder, graph.root());
        let bound = builder.finish_one(root);
        assert_eq!(
            bound.root().retired_axis(),
            None,
            "bind_retired_axes left a reachable retired axis"
        );
        bound
    }

    /// Binding rewrites the reachable graph and removes retired variables from
    /// the graph presented to code generation.
    #[test]
    fn substitution_clears_reachable_retired_axes() {
        let mut builder = ExprBuilder::new();
        let y = builder.var(1);
        let z = builder.var(2);
        let w = builder.var(3);
        let zw = builder.binary(OpKind::Add, z, w);
        let root = builder.binary(OpKind::Mul, y, zw);
        let graph = builder.finish_one(root);
        assert!(
            graph
                .root()
                .retired_axis()
                .is_some_and(|v| v == 2 || v == 3),
            "the fixture names a retired axis (which one depends on walk order)"
        );

        let bound = bind_retired_axes(&graph);
        assert_eq!(
            bound.root().retired_axis(),
            None,
            "nothing reachable names a retired axis after substitution"
        );
    }

    #[test]
    fn parse_kernel_code_basic() {
        // Simple expression: no structural sharing expected.
        let graph = parse_kernel_code("(X + Y)").unwrap();
        assert!(
            graph.root().dag().len() >= 3,
            "expected at least 3 nodes (X, Y, Add); got {}",
            graph.root().dag().len()
        );
        let _ = graph.root(); // root is valid
    }

    #[test]
    fn parse_kernel_code_structural_sharing() {
        // (X + X): the two X leaves are structurally identical and should share an id.
        let graph = parse_kernel_code("(X + X)").unwrap();
        // Without sharing: 3 nodes (X, X, Add). With sharing: 2 nodes (X, Add).
        assert_eq!(
            graph.root().dag().len(),
            2,
            "expected 2 unique nodes for (X + X) with sharing, got {}",
            graph.root().dag().len()
        );
    }

    #[test]
    fn parse_kernel_code_deeply_shared() {
        // ((X + Y) * (X + Y)): the (X + Y) subtree appears twice — should be shared.
        // Without sharing: 7 nodes. With sharing: 4 nodes (X, Y, Add, Mul).
        let graph = parse_kernel_code("((X + Y) * (X + Y))").unwrap();
        assert_eq!(
            graph.root().dag().len(),
            4,
            "expected 4 unique nodes for ((X+Y)*(X+Y)) with sharing, got {}",
            graph.root().dag().len()
        );
    }

    #[test]
    fn parse_kernel_code_round_trip() {
        // Graph parse + graph_to_kernel_code should re-parse cleanly.
        let src = "((X * Y) + (X * Y))";
        let graph = parse_kernel_code(src).unwrap();
        let code = graph_to_kernel_code(&graph);
        assert!(
            parse_kernel_code(&code).is_some(),
            "graph round-trip produced un-parseable code: {code}"
        );
    }

    #[test]
    fn parse_kernel_code_methods() {
        let src = "(((X).abs()).min(Y)).mul_add(Z, W)";
        let graph = parse_kernel_code(src).unwrap();
        let code = graph_to_kernel_code(&graph);
        let reparsed = parse_kernel_code(&code).unwrap();
        assert_eq!(code, graph_to_kernel_code(&reparsed));
        assert!(
            graph.root().dag().len() >= 7,
            "expected method-heavy parse to build a real DAG"
        );
    }

    // ========================================================================
    // Kernel Code Parser Tests
    // ========================================================================

    // ================================================================
    // expr_to_kernel_code round-trip tests
    // ================================================================
}
