//! # Expression parsing and serialization utilities for NNUE training.
//!
//! Provides parsers for two expression syntaxes:
//! - **S-expression**: `Add(Mul(Var(0), Var(1)), Var(2))` (test-only repro format)
//! - **Kernel code**: `(X * Y) + Z` (human-readable, round-trips with
//!   `parse_kernel_code`/`term_to_kernel_code`)

use std::collections::HashMap;

use pixelflow_ir::{
    EmitStyle, Environment, ExprBuilder, ExprData, ExprRef, Node, OpKind, Rooted, Term,
};

// ============================================================================
// Expression Parsing (for loading training data)
// ============================================================================

/// Parse an expression from a string representation.
///
/// Test-only helper for older logged repros in `OpName(child1, child2, ...)`
/// form such as `Add(Mul(Var(0), Var(1)), Var(2))`.
#[cfg(test)]
pub fn parse_expr(s: &str) -> Option<(Rooted<ExprData>, Environment)> {
    let mut arena = ExprBuilder::new();
    let root = parse_expr_into(s, &mut arena)?;
    Some(arena.finish(&[root]))
}

/// Recursive S-expression parser that builds directly into an [`ExprBuilder`].
#[cfg(test)]
fn parse_expr_into(s: &str, arena: &mut ExprBuilder) -> Option<ExprRef> {
    let s = s.trim();

    if let Some(inner) = s.strip_prefix("Var(").and_then(|r| r.strip_suffix(')')) {
        let idx: u8 = inner.trim().parse().ok()?;
        return Some(arena.push_var(idx));
    }
    if let Some(inner) = s.strip_prefix("Const(").and_then(|r| r.strip_suffix(')')) {
        let val: f32 = inner.trim().parse().ok()?;
        return Some(arena.push_const(val));
    }

    let paren_pos = s.find('(')?;
    let op_name = s[..paren_pos].to_lowercase();
    let inner = &s[paren_pos + 1..s.len() - 1];
    let children = split_args(inner);

    // `fract` and `hypot` are compound ops (no hardware instruction backs
    // them, see pixelflow-ir/src/kernel.rs), so they no longer have a single
    // OpKind to construct via push_unary/push_binary below. Build the same
    // primitive subgraph pixelflow_ir::backend::compounds::Compounds does.
    match op_name.as_str() {
        "fract" if children.len() == 1 => {
            // fract(x) = x - floor(x)
            let a = parse_expr_into(children[0], arena)?;
            let floor_a = arena.push_unary(OpKind::Floor, a);
            return Some(arena.push_binary(OpKind::Sub, a, floor_a));
        }
        "hypot" if children.len() == 2 => {
            // hypot(x, y) = sqrt(x*x + y*y)
            let a = parse_expr_into(children[0], arena)?;
            let b = parse_expr_into(children[1], arena)?;
            let aa = arena.push_binary(OpKind::Mul, a, a);
            let bb = arena.push_binary(OpKind::Mul, b, b);
            let sum = arena.push_binary(OpKind::Add, aa, bb);
            return Some(arena.push_unary(OpKind::Sqrt, sum));
        }
        _ => {}
    }

    let op = parse_op_kind(&op_name)?;
    match (op.arity(), children.len()) {
        (1, 1) => {
            let a = parse_expr_into(children[0], arena)?;
            Some(arena.push_unary(op, a))
        }
        (2, 2) => {
            let a = parse_expr_into(children[0], arena)?;
            let b = parse_expr_into(children[1], arena)?;
            Some(arena.push_binary(op, a, b))
        }
        (3, 3) => {
            let a = parse_expr_into(children[0], arena)?;
            let b = parse_expr_into(children[1], arena)?;
            let c = parse_expr_into(children[2], arena)?;
            Some(arena.push_ternary(op, a, b, c))
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

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum NodeKey {
    Var(u8),
    Const(u32),
    Unary(OpKind, ExprRef),
    Binary(OpKind, ExprRef, ExprRef),
    Ternary(OpKind, ExprRef, ExprRef, ExprRef),
}

struct ArenaInterner {
    arena: ExprBuilder,
    nodes: HashMap<NodeKey, ExprRef>,
}

impl ArenaInterner {
    fn new() -> Self {
        Self {
            arena: ExprBuilder::new(),
            nodes: HashMap::new(),
        }
    }

    fn push_key(&mut self, key: NodeKey) -> ExprRef {
        if let Some(&existing) = self.nodes.get(&key) {
            return existing;
        }

        let id = match key {
            NodeKey::Var(i) => self.arena.push_var(i),
            NodeKey::Const(bits) => self.arena.push_const(f32::from_bits(bits)),
            NodeKey::Unary(op, a) => self.arena.push_unary(op, a),
            NodeKey::Binary(op, a, b) => self.arena.push_binary(op, a, b),
            NodeKey::Ternary(op, a, b, c) => self.arena.push_ternary(op, a, b, c),
        };
        self.nodes.insert(key, id);
        id
    }

    fn push_var(&mut self, index: u8) -> ExprRef {
        self.push_key(NodeKey::Var(index))
    }

    fn push_const(&mut self, value: f32) -> ExprRef {
        self.push_key(NodeKey::Const(value.to_bits()))
    }

    fn push_unary(&mut self, op: OpKind, a: ExprRef) -> ExprRef {
        self.push_key(NodeKey::Unary(op, a))
    }

    fn push_binary(&mut self, op: OpKind, a: ExprRef, b: ExprRef) -> ExprRef {
        self.push_key(NodeKey::Binary(op, a, b))
    }

    fn push_ternary(&mut self, op: OpKind, a: ExprRef, b: ExprRef, c: ExprRef) -> ExprRef {
        self.push_key(NodeKey::Ternary(op, a, b, c))
    }

    fn finish(self, root: ExprRef) -> (Rooted<ExprData>, Environment) {
        self.arena.finish(&[root])
    }
}

enum ParseOp {
    PrefixNeg,
    Binary(OpKind),
    GroupParen,
    GroupMethod { op: OpKind, commas: usize },
}

struct ArenaKernelParser<'a> {
    input: &'a str,
    pos: usize,
    interner: ArenaInterner,
}

impl<'a> ArenaKernelParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            pos: 0,
            interner: ArenaInterner::new(),
        }
    }

    fn parse(mut self) -> Option<(Rooted<ExprData>, Environment)> {
        let root = self.parse_expr()?;
        self.skip_ws();
        (self.pos == self.input.len()).then(|| self.interner.finish(root))
    }

    fn parse_expr(&mut self) -> Option<ExprRef> {
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
                        values.push(self.interner.push_var(var));
                        expecting_operand = false;
                    }
                    c if c.is_ascii_digit() || c == '.' => {
                        let number = self.parse_number_literal()?;
                        values.push(self.interner.push_const(number));
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
                    values.push(self.interner.push_unary(op, base));
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
                                    self.interner.push_binary(op, a, b)
                                }
                                3 => {
                                    let c = values.pop()?;
                                    let b = values.pop()?;
                                    let a = values.pop()?;
                                    self.interner.push_ternary(op, a, b, c)
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
        interner: &mut ArenaInterner,
        values: &mut Vec<ExprRef>,
        ops: &mut Vec<ParseOp>,
    ) -> Option<()> {
        while matches!(ops.last(), Some(ParseOp::PrefixNeg)) {
            ops.pop();
            let value = values.pop()?;
            values.push(interner.push_unary(OpKind::Neg, value));
        }
        Some(())
    }

    fn reduce_binary_ops(
        interner: &mut ArenaInterner,
        values: &mut Vec<ExprRef>,
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
                    values.push(interner.push_binary(current, a, b));
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
        interner: &mut ArenaInterner,
        values: &mut Vec<ExprRef>,
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
                    values.push(interner.push_binary(op, a, b));
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

/// Parser result: (parsed value, remaining input)
///
/// Parse kernel code directly into a DAG with structural sharing.
///
/// Identical subexpressions map to the same node, so the result is a true DAG
/// rather than a duplicated tree. The dedup key is `(op, child refs)`: two
/// nodes are shared iff they have the same [`OpKind`] and the same children
/// (or the same leaf value).
///
/// Returns `None` if the input fails to parse.
pub fn parse_kernel_code(s: &str) -> Option<(Rooted<ExprData>, Environment)> {
    ArenaKernelParser::new(s.trim()).parse()
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
// Arena → Kernel Code Serialization
// ============================================================================

/// Convert a term into kernel code syntax.
pub fn term_to_kernel_code(term: Term<'_>) -> String {
    enum Task<'a> {
        Visit(Node<'a, ExprData>),
        Emit {
            node: Node<'a, ExprData>,
            arity: usize,
        },
    }

    let mut stack = vec![Task::Visit(term.root())];
    let mut result_stack: Vec<String> = Vec::new();

    while let Some(task) = stack.pop() {
        match task {
            Task::Visit(node) => {
                stack.push(Task::Emit {
                    node,
                    arity: node.child_count(),
                });
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
                    // Z and W are retired *axes*, but the frozen corpus
                    // fixtures this format round-trips still name them, and
                    // the parser still reads them. Printing them keeps the
                    // round trip total; binding them to constants before
                    // evaluation is `bind_retired_axes`'s job, not this one's.
                    ExprData::Var(2) => "Z".into(),
                    ExprData::Var(3) => "W".into(),
                    ExprData::Var(i) => {
                        panic!("term_to_kernel_code: variable index {i} exceeds the X/Y/Z/W range")
                    }
                    ExprData::Const(bits) => format_const_kc(f32::from_bits(bits)),
                    ExprData::Param(i) => {
                        panic!("Param({i}) reached term_to_kernel_code — substitute params first")
                    }
                    ExprData::Buffer(b) => panic!(
                        "Buffer({}) reached term_to_kernel_code — memory ops require \
                         a binding table, not yet wired (M2, see KERNELS_AND_LATTICES.md)",
                        b.0
                    ),
                    ExprData::Uniform(u) => panic!(
                        "Uniform({}) reached term_to_kernel_code — kernel code has \
                         no block to read it from",
                        u.0
                    ),
                    ExprData::Op(op) => {
                        assert!(
                            arity <= 3,
                            "term_to_kernel_code: {} with {arity} children is not \
                             representable in kernel code syntax",
                            op.name()
                        );
                        emit_op_kc(op, &args)
                    }
                };
                result_stack.push(emitted);
            }
        }
    }

    result_stack
        .pop()
        .unwrap_or_else(|| panic!("term_to_kernel_code: empty result stack"))
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
        panic!("arena_to_kernel_code: non-finite constant {v} cannot be represented");
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
    #[cfg(target_arch = "aarch64")]
    use crate::jit_bench::benchmark_jit_arena;

    const REWRITE_BUG_INPUTS: [f32; 2] = [0.5, 0.7];

    /// The values the frozen corpus expressions' third and fourth variables
    /// take. Those were the Z and W coordinates; a lattice has two axes now,
    /// so the fixtures' `Var(2)`/`Var(3)` are bound as arguments — invariant
    /// across the lattice, which at a single point is the same number the
    /// coordinate carried.
    const REWRITE_BUG_ARGS: [f32; 2] = [1.3, -0.2];

    // The scalar reference is the differential-testing oracle: it runs the same
    // `expand_transcendentals` lowering the JIT runs, so Pow/exp/log evaluate
    // through their one definition rather than a second host-libm semantics.
    /// Replace the frozen fixtures' `Var(2)`/`Var(3)` with the values they
    /// used to carry as the Z and W coordinates.
    ///
    /// **Every** consumer of a fixture goes through this, which is the whole
    /// point. The scalar oracle used to substitute here while the JIT
    /// harness fed the emitter zeros in those lanes, so the two evaluated
    /// different programs and disagreed by three orders of magnitude — a
    /// silent numeric divergence, not a failure, and invisible on x86 where
    /// the JIT half of the comparison is not even compiled. `emit::compile`
    /// now refuses an arena that names a retired axis, so a fixture that
    /// skipped this would panic rather than diverge.
    ///
    /// Constants, not uniforms: `benchmark_jit_arena` calls the collapse ABI
    /// with a null context, so a uniform read would fault. A constant needs
    /// no context and denotes exactly the same number on both sides.
    fn bind_retired_axes(term: Term<'_>) -> (Rooted<ExprData>, Environment) {
        let root = term.root();
        let mut out = ExprBuilder::new();
        let mut map = term.dag().side_table(None);
        let mut work = vec![(root, false)];
        while let Some((node, expanded)) = work.pop() {
            if map[node].is_some() {
                continue;
            }
            if !expanded {
                work.push((node, true));
                for child in node.children() {
                    if map[child].is_none() {
                        work.push((child, false));
                    }
                }
                continue;
            }
            let built = match *node {
                ExprData::Var(i) if pixelflow_ir::RETIRED_COORD_AXES.contains(&i) => {
                    let slot = usize::from(i) - pixelflow_ir::COORD_AXES;
                    out.push_const(REWRITE_BUG_ARGS[slot])
                }
                ExprData::Var(i) => out.push_var(i),
                ExprData::Const(bits) => out.push_const(f32::from_bits(bits)),
                ExprData::Param(i) => out.push_param(i),
                ExprData::Op(op) => {
                    let kids: Vec<ExprRef> = node
                        .children()
                        .map(|c| map[c].expect("child rebuilt before parent"))
                        .collect();
                    out.push_nary(op, &kids)
                }
                other => panic!("bind_retired_axes: fixtures never hold {other:?}"),
            };
            map[node] = Some(built);
        }
        let new_root = map[root].expect("the root is its own descendant");
        let (rebuilt, env) = out.finish(&[new_root]);
        // The precondition `emit::compile` will assert, checked here where
        // *every* caller reaches it on *every* architecture. The scalar/JIT
        // comparisons below are `cfg(target_arch = "aarch64")`, so on an x86
        // machine they do not exist and the emitter is never reached with
        // these graphs at all — this file has learned that from a macOS
        // runner twice. Asserting the precondition rather than the outcome is
        // what makes the next miss local.
        assert_eq!(
            pixelflow_ir::retired_axis(rebuilt.entry()),
            None,
            "bind_retired_axes left a reachable retired axis"
        );
        (rebuilt, env)
    }

    fn eval_scalar_bound(term: Term<'_>, vars: &[f32; 2]) -> f32 {
        let (bound, env) = bind_retired_axes(term);
        pixelflow_ir::eval_scalar(
            Term::new(bound.entry(), &env),
            vars,
            &pixelflow_ir::BindingTable::empty(),
        )
    }

    fn logged_expr_scalar_output(src: &str) -> f32 {
        let (expr, env) = parse_expr(src).unwrap_or_else(|| panic!("parse_expr failed: {src}"));
        eval_scalar_bound(Term::new(expr.entry(), &env), &REWRITE_BUG_INPUTS)
    }

    #[cfg(target_arch = "aarch64")]
    fn logged_expr_jit_output(src: &str) -> f32 {
        let (expr, env) = parse_expr(src).unwrap_or_else(|| panic!("parse_expr failed: {src}"));
        // The same substitution the oracle applies. Skipping it here is the
        // bug this pairing exists to catch.
        let (bound, benv) = bind_retired_axes(Term::new(expr.entry(), &env));
        benchmark_jit_term(Term::new(bound.entry(), &benv))
            .unwrap_or_else(|err| panic!("benchmark_jit_term failed for {src}: {err:?}"))
            .output[0]
    }

    fn logged_expr_roundtrip_scalar_output(src: &str) -> f32 {
        // arena -> kernel-code -> arena round-trip must preserve scalar semantics.
        let (expr, env) = parse_expr(src).unwrap_or_else(|| panic!("parse_expr failed: {src}"));
        let kernel = term_to_kernel_code(Term::new(expr.entry(), &env));
        let (re_expr, re_env) = parse_kernel_code(&kernel)
            .unwrap_or_else(|| panic!("parse_kernel_code failed: {kernel}"));
        eval_scalar_bound(Term::new(re_expr.entry(), &re_env), &REWRITE_BUG_INPUTS)
    }

    #[cfg(target_arch = "aarch64")]
    fn assert_scalar_and_jit_close(src: &str, epsilon: f32) {
        let scalar = logged_expr_scalar_output(src);
        let jit = logged_expr_jit_output(src);
        let diff = (scalar - jit).abs();
        assert!(
            diff <= epsilon,
            "scalar/JIT mismatch\nexpr: {src}\nscalar: {scalar}\njit: {jit}\ndiff: {diff} > {epsilon}"
        );
    }

    /// Substitution rewrites the reachable graph; what matters — and what
    /// the emitter asks — is whether a retired axis is reachable from the
    /// root. Scanning every node instead cost this change a CI round trip.
    #[test]
    fn substitution_clears_the_reachable_retired_axes() {
        let mut b = ExprBuilder::new();
        let y = b.push_var(1);
        let z = b.push_var(2);
        let w = b.push_var(3);
        let zw = b.push_binary(pixelflow_ir::OpKind::Add, z, w);
        let root = b.push_binary(pixelflow_ir::OpKind::Mul, y, zw);
        let (rooted, env) = b.finish(&[root]);
        assert!(
            pixelflow_ir::retired_axis(rooted.entry()).is_some_and(|v| v == 2 || v == 3),
            "the fixture names a retired axis (which one depends on walk order)"
        );

        let (bound, _benv) = bind_retired_axes(Term::new(rooted.entry(), &env));
        assert_eq!(
            pixelflow_ir::retired_axis(bound.entry()),
            None,
            "nothing reachable names a retired axis after substitution"
        );
    }

    #[test]
    fn parse_kernel_code_basic() {
        // Simple expression: no structural sharing expected.
        let (expr, env) = parse_kernel_code("(X + Y)").unwrap();
        assert!(
            expr.len() >= 3,
            "expected at least 3 nodes (X, Y, Add); got {}",
            expr.len()
        );
        let _ = env; // the graph parsed
    }

    #[test]
    fn parse_kernel_code_structural_sharing() {
        // (X + X): the two X leaves are structurally identical and should share an id.
        let (expr, _env) = parse_kernel_code("(X + X)").unwrap();
        // Without sharing: 3 nodes (X, X, Add). With sharing: 2 nodes (X, Add).
        assert_eq!(
            expr.len(),
            2,
            "expected 2 unique nodes for (X + X) with sharing, got {}",
            expr.len()
        );
    }

    #[test]
    fn parse_kernel_code_deeply_shared() {
        // ((X + Y) * (X + Y)): the (X + Y) subtree appears twice — should be shared.
        // Without sharing: 7 nodes. With sharing: 4 nodes (X, Y, Add, Mul).
        let (expr, _env) = parse_kernel_code("((X + Y) * (X + Y))").unwrap();
        assert_eq!(
            expr.len(),
            4,
            "expected 4 unique nodes for ((X+Y)*(X+Y)) with sharing, got {}",
            expr.len()
        );
    }

    #[test]
    fn parse_kernel_code_round_trip() {
        // Arena parse + arena_to_kernel_code should re-parse cleanly.
        let src = "((X * Y) + (X * Y))";
        let (expr, env) = parse_kernel_code(src).unwrap();
        let code = term_to_kernel_code(Term::new(expr.entry(), &env));
        assert!(
            parse_kernel_code(&code).is_some(),
            "arena round-trip produced un-parseable code: {code}"
        );
    }

    #[test]
    fn parse_kernel_code_methods() {
        let src = "(((X).abs()).min(Y)).mul_add(Z, W)";
        let (expr, env) = parse_kernel_code(src).unwrap();
        let code = term_to_kernel_code(Term::new(expr.entry(), &env));
        let (reparsed, re_env) = parse_kernel_code(&code).unwrap();
        assert_eq!(
            code,
            term_to_kernel_code(Term::new(reparsed.entry(), &re_env))
        );
        assert!(
            expr.len() >= 7,
            "expected method-heavy parse to build a real DAG"
        );
    }

    #[test]
    fn parse_expr_fract_builds_sub_floor_compound() {
        // fract(x) = x - floor(x); no OpKind::Fract exists (see kernel.rs) so
        // this must expand to the primitive subgraph, not a single node.
        let src = "fract(Var(0))";
        let (expr, env) = parse_expr(src).unwrap_or_else(|| panic!("parse failed: {src}"));
        assert_eq!(
            eval_scalar_bound(Term::new(expr.entry(), &env), &[1.75, 0.0]),
            0.75
        );
        assert_eq!(
            eval_scalar_bound(Term::new(expr.entry(), &env), &[-1.25, 0.0]),
            0.75
        );
    }

    #[test]
    fn parse_expr_hypot_builds_sqrt_mul_add_compound() {
        // hypot(x, y) = sqrt(x*x + y*y); no OpKind::Hypot exists (see kernel.rs).
        let src = "hypot(Var(0), Var(1))";
        let (expr, env) = parse_expr(src).unwrap_or_else(|| panic!("parse failed: {src}"));
        assert_eq!(
            eval_scalar_bound(Term::new(expr.entry(), &env), &[3.0, 4.0]),
            5.0
        );
    }

    // ========================================================================
    // Kernel Code Parser Tests
    // ========================================================================

    // ================================================================
    // expr_to_kernel_code round-trip tests
    // ================================================================

    #[test]
    fn seed_42_t1_pair_is_close_under_scalar_semantics() {
        let initial = "log2(add(abs(neg(atan2(pow(add(abs(neg(pow(add(abs(neg(neg(Const(-0.9631642)))), Const(0.001)), Const(-1)))), Const(0.001)), Const(-1)), mul(exp(add(pow(add(abs(neg(neg(atan2(Const(0.1167655), Var(1))))), Const(0.001)), Const(-1)), add(min(Var(3), Var(1)), log2(add(abs(neg(Var(3))), Const(0.001)))))), min(min(sub(cos(neg(Const(1.5691397))), add(Const(-1.1460416), Var(2))), abs(exp(Var(0)))), log10(add(abs(min(sub(Const(0.1855638), Var(1)), exp(Var(1)))), Const(0.001)))))))), Const(0.001)))";
        let final_ = "log2(add(Const(0.001), abs(atan2(pow(add(abs(neg(pow(add(abs(Const(-0.9631642)), Const(0.001)), Const(-1)))), Const(0.001)), Const(-1)), mul(mul(min(min(sub(cos(neg(Const(1.5691397))), add(Const(-1.1460416), Var(2))), abs(exp(Var(0)))), log10(add(abs(min(sub(Const(0.1855638), Var(1)), exp(Var(1)))), Const(0.001)))), exp(pow(add(abs(atan2(Const(0.1167655), Var(1))), Const(0.001)), Const(-1)))), exp(add(min(Var(3), Var(1)), log2(add(abs(neg(Var(3))), Const(0.001))))))))))";
        let initial = logged_expr_scalar_output(initial);
        let final_ = logged_expr_scalar_output(final_);
        let diff = (initial - final_).abs();
        assert!(
            diff <= 1e-6,
            "scalar rewrite mismatch\ninitial: {initial}\nfinal: {final_}\ndiff: {diff}"
        );
    }

    #[test]
    // These deeply-nested expressions exceed the x86-64 JIT's spill-free SSE
    // register budget (it rejects them with an error). The aarch64 backend uses
    // linear-scan allocation with spilling, so it handles arbitrary depth.
    #[cfg(target_arch = "aarch64")]
    fn seed_42_t1_initial_jit_matches_scalar() {
        let initial = "log2(add(abs(neg(atan2(pow(add(abs(neg(pow(add(abs(neg(neg(Const(-0.9631642)))), Const(0.001)), Const(-1)))), Const(0.001)), Const(-1)), mul(exp(add(pow(add(abs(neg(neg(atan2(Const(0.1167655), Var(1))))), Const(0.001)), Const(-1)), add(min(Var(3), Var(1)), log2(add(abs(neg(Var(3))), Const(0.001)))))), min(min(sub(cos(neg(Const(1.5691397))), add(Const(-1.1460416), Var(2))), abs(exp(Var(0)))), log10(add(abs(min(sub(Const(0.1855638), Var(1)), exp(Var(1)))), Const(0.001)))))))), Const(0.001)))";
        assert_scalar_and_jit_close(initial, 1e-3);
    }

    #[test]
    fn seed_42_t1_initial_roundtrip_scalar_matches_tree_scalar() {
        let initial = "log2(add(abs(neg(atan2(pow(add(abs(neg(pow(add(abs(neg(neg(Const(-0.9631642)))), Const(0.001)), Const(-1)))), Const(0.001)), Const(-1)), mul(exp(add(pow(add(abs(neg(neg(atan2(Const(0.1167655), Var(1))))), Const(0.001)), Const(-1)), add(min(Var(3), Var(1)), log2(add(abs(neg(Var(3))), Const(0.001)))))), min(min(sub(cos(neg(Const(1.5691397))), add(Const(-1.1460416), Var(2))), abs(exp(Var(0)))), log10(add(abs(min(sub(Const(0.1855638), Var(1)), exp(Var(1)))), Const(0.001)))))))), Const(0.001)))";
        let tree = logged_expr_scalar_output(initial);
        let roundtrip = logged_expr_roundtrip_scalar_output(initial);
        assert!(
            (tree - roundtrip).abs() <= 1e-6,
            "tree: {tree}, roundtrip: {roundtrip}"
        );
    }

    #[test]
    // These deeply-nested expressions exceed the x86-64 JIT's spill-free SSE
    // register budget (it rejects them with an error). The aarch64 backend uses
    // linear-scan allocation with spilling, so it handles arbitrary depth.
    #[cfg(target_arch = "aarch64")]
    fn seed_42_t1_final_jit_matches_scalar() {
        let final_ = "log2(add(Const(0.001), abs(atan2(pow(add(abs(neg(pow(add(abs(Const(-0.9631642)), Const(0.001)), Const(-1)))), Const(0.001)), Const(-1)), mul(mul(min(min(sub(cos(neg(Const(1.5691397))), add(Const(-1.1460416), Var(2))), abs(exp(Var(0)))), log10(add(abs(min(sub(Const(0.1855638), Var(1)), exp(Var(1)))), Const(0.001)))), exp(pow(add(abs(atan2(Const(0.1167655), Var(1))), Const(0.001)), Const(-1)))), exp(add(min(Var(3), Var(1)), log2(add(abs(neg(Var(3))), Const(0.001))))))))))";
        assert_scalar_and_jit_close(final_, 1e-3);
    }

    #[test]
    fn seed_24042_t52_pair_is_close_under_scalar_semantics() {
        let initial = "div(min(add(mul(min(pow(add(abs(neg(neg(abs(neg(neg(add(mul(cos(neg(neg(Var(1)))), Const(0.52093434)), Const(-1.414685)))))))), Const(0.001)), Const(-1)), Var(1)), log10(add(abs(neg(neg(add(add(max(add(mul(Var(2), Var(2)), Var(3)), Var(1)), atan2(atan2(Var(0), Var(3)), add(Var(1), neg(Var(3))))), mul(ln(add(abs(neg(neg(neg(Const(0.12621832))))), Const(0.001))), mul_add(Var(0), cos(neg(neg(Var(3)))), pow(add(abs(neg(neg(Const(-0.20414245)))), Const(0.001)), Const(-0.5)))))))), Const(0.001)))), pow(add(abs(neg(neg(atan2(Var(2), log10(add(abs(neg(neg(tan(Var(1))))), Const(0.001))))))), Const(0.001)), Const(-0.5))), log2(add(abs(neg(neg(Var(0)))), Const(0.001)))), add(abs(neg(pow(add(abs(neg(pow(add(abs(neg(log2(add(abs(neg(mul(min(add(mul(log10(add(abs(neg(neg(Const(1.2403846)))), Const(0.001))), Var(2)), mul(log10(add(abs(neg(neg(Const(1.2403846)))), Const(0.001))), Var(3))), max(ln(add(abs(neg(neg(Const(0.2514913)))), Const(0.001))), mul(Var(0), Const(0.5072496)))), mul(pow(abs(neg(neg(log10(add(abs(neg(neg(Var(3)))), Const(0.001)))))), Const(0.5)), ln(add(abs(neg(sin(Var(2)))), Const(0.001))))))), Const(0.001))))), Const(0.001)), mul(add(add(mul(div(add(Const(0.61049294), Const(1.354384)), add(abs(neg(mul(Var(0), Var(0)))), Const(0.001))), mul(tan(Const(-1.6860065)), recip(add(abs(neg(Const(-1.7208018))), Const(0.001))))), add(mul(max(Var(3), Var(0)), cos(neg(Var(0)))), mul_add(Var(3), Var(1), Var(0)))), log10(add(abs(neg(Const(-1.1988422))), Const(0.001)))), max(abs(neg(ln(add(abs(neg(pow(add(abs(Var(1)), Const(0.001)), Var(2)))), Const(0.001))))), ln(add(abs(mul(Const(-0.47071946), pow(add(abs(neg(Const(-1.670574))), Const(0.001)), Var(0)))), Const(0.001)))))))), Const(0.001)), Const(-1)))), Const(0.001)))";
        let final_ = "div(min(add(mul(min(pow(add(abs(neg(neg(abs(neg(neg(add(mul(cos(neg(neg(Var(1)))), Const(0.52093434)), Const(-1.414685)))))))), Const(0.001)), Const(-1)), Var(1)), log10(add(abs(neg(neg(add(add(max(add(mul(Var(2), Var(2)), Var(3)), Var(1)), atan2(atan2(Var(0), Var(3)), add(Var(1), neg(Var(3))))), mul(ln(add(Const(0.12621832), Const(0.001))), mul_add(Var(0), cos(neg(neg(Var(3)))), pow(add(Const(0.20414245), Const(0.001)), Const(-0.5)))))))), Const(0.001)))), pow(add(abs(neg(neg(atan2(Var(2), log10(add(abs(neg(neg(tan(Var(1))))), Const(0.001))))))), Const(0.001)), Const(-0.5))), log2(add(abs(neg(neg(Var(0)))), Const(0.001)))), add(abs(neg(pow(add(abs(neg(pow(add(abs(neg(log2(add(abs(neg(mul(min(add(mul(Const(0.093906365), Var(2)), mul(Const(0.093906365), Var(3))), max(Const(-1.3763785), mul(Var(0), Const(0.5072496)))), mul(pow(abs(neg(neg(log10(add(abs(neg(neg(Var(3)))), Const(0.001)))))), Const(0.5)), ln(add(abs(neg(sin(Var(2)))), Const(0.001))))))), Const(0.001))))), Const(0.001)), mul(add(add(mul(div(add(Const(0.61049294), Const(1.354384)), add(abs(neg(mul(Var(0), Var(0)))), Const(0.001))), mul(Const(8.641348), recip(add(Const(1.7208018), Const(0.001))))), add(mul(max(Var(3), Var(0)), cos(neg(Var(0)))), mul_add(Var(3), Var(1), Var(0)))), log10(add(Const(1.1988422), Const(0.001)))), max(abs(neg(ln(add(abs(neg(pow(add(abs(Var(1)), Const(0.001)), Var(2)))), Const(0.001))))), ln(add(abs(mul(Const(-0.47071946), pow(add(Const(1.670574), Const(0.001)), Var(0)))), Const(0.001)))))))), Const(0.001)), Const(-1)))), Const(0.001)))";
        let initial = logged_expr_scalar_output(initial);
        let final_ = logged_expr_scalar_output(final_);
        let diff = (initial - final_).abs();
        assert!(
            diff <= 1e-3,
            "scalar rewrite mismatch\ninitial: {initial}\nfinal: {final_}\ndiff: {diff}"
        );
    }

    #[test]
    // These deeply-nested expressions exceed the x86-64 JIT's spill-free SSE
    // register budget (it rejects them with an error). The aarch64 backend uses
    // linear-scan allocation with spilling, so it handles arbitrary depth.
    #[cfg(target_arch = "aarch64")]
    fn seed_24042_t52_initial_jit_matches_scalar() {
        let initial = "div(min(add(mul(min(pow(add(abs(neg(neg(abs(neg(neg(add(mul(cos(neg(neg(Var(1)))), Const(0.52093434)), Const(-1.414685)))))))), Const(0.001)), Const(-1)), Var(1)), log10(add(abs(neg(neg(add(add(max(add(mul(Var(2), Var(2)), Var(3)), Var(1)), atan2(atan2(Var(0), Var(3)), add(Var(1), neg(Var(3))))), mul(ln(add(abs(neg(neg(neg(Const(0.12621832))))), Const(0.001))), mul_add(Var(0), cos(neg(neg(Var(3)))), pow(add(abs(neg(neg(Const(-0.20414245)))), Const(0.001)), Const(-0.5)))))))), Const(0.001)))), pow(add(abs(neg(neg(atan2(Var(2), log10(add(abs(neg(neg(tan(Var(1))))), Const(0.001))))))), Const(0.001)), Const(-0.5))), log2(add(abs(neg(neg(Var(0)))), Const(0.001)))), add(abs(neg(pow(add(abs(neg(pow(add(abs(neg(log2(add(abs(neg(mul(min(add(mul(log10(add(abs(neg(neg(Const(1.2403846)))), Const(0.001))), Var(2)), mul(log10(add(abs(neg(neg(Const(1.2403846)))), Const(0.001))), Var(3))), max(ln(add(abs(neg(neg(Const(0.2514913)))), Const(0.001))), mul(Var(0), Const(0.5072496)))), mul(pow(abs(neg(neg(log10(add(abs(neg(neg(Var(3)))), Const(0.001)))))), Const(0.5)), ln(add(abs(neg(sin(Var(2)))), Const(0.001))))))), Const(0.001))))), Const(0.001)), mul(add(add(mul(div(add(Const(0.61049294), Const(1.354384)), add(abs(neg(mul(Var(0), Var(0)))), Const(0.001))), mul(tan(Const(-1.6860065)), recip(add(abs(neg(Const(-1.7208018))), Const(0.001))))), add(mul(max(Var(3), Var(0)), cos(neg(Var(0)))), mul_add(Var(3), Var(1), Var(0)))), log10(add(abs(neg(Const(-1.1988422))), Const(0.001)))), max(abs(neg(ln(add(abs(neg(pow(add(abs(Var(1)), Const(0.001)), Var(2)))), Const(0.001))))), ln(add(abs(mul(Const(-0.47071946), pow(add(abs(neg(Const(-1.670574))), Const(0.001)), Var(0)))), Const(0.001)))))))), Const(0.001)), Const(-1)))), Const(0.001)))";
        assert_scalar_and_jit_close(initial, 1e-3);
    }

    #[test]
    fn seed_24042_t52_initial_roundtrip_scalar_matches_tree_scalar() {
        let initial = "div(min(add(mul(min(pow(add(abs(neg(neg(abs(neg(neg(add(mul(cos(neg(neg(Var(1)))), Const(0.52093434)), Const(-1.414685)))))))), Const(0.001)), Const(-1)), Var(1)), log10(add(abs(neg(neg(add(add(max(add(mul(Var(2), Var(2)), Var(3)), Var(1)), atan2(atan2(Var(0), Var(3)), add(Var(1), neg(Var(3))))), mul(ln(add(abs(neg(neg(neg(Const(0.12621832))))), Const(0.001))), mul_add(Var(0), cos(neg(neg(Var(3)))), pow(add(abs(neg(neg(Const(-0.20414245)))), Const(0.001)), Const(-0.5)))))))), Const(0.001)))), pow(add(abs(neg(neg(atan2(Var(2), log10(add(abs(neg(neg(tan(Var(1))))), Const(0.001))))))), Const(0.001)), Const(-0.5))), log2(add(abs(neg(neg(Var(0)))), Const(0.001)))), add(abs(neg(pow(add(abs(neg(pow(add(abs(neg(log2(add(abs(neg(mul(min(add(mul(log10(add(abs(neg(neg(Const(1.2403846)))), Const(0.001))), Var(2)), mul(log10(add(abs(neg(neg(Const(1.2403846)))), Const(0.001))), Var(3))), max(ln(add(abs(neg(neg(Const(0.2514913)))), Const(0.001))), mul(Var(0), Const(0.5072496)))), mul(pow(abs(neg(neg(log10(add(abs(neg(neg(Var(3)))), Const(0.001)))))), Const(0.5)), ln(add(abs(neg(sin(Var(2)))), Const(0.001))))))), Const(0.001))))), Const(0.001)), mul(add(add(mul(div(add(Const(0.61049294), Const(1.354384)), add(abs(neg(mul(Var(0), Var(0)))), Const(0.001))), mul(tan(Const(-1.6860065)), recip(add(abs(neg(Const(-1.7208018))), Const(0.001))))), add(mul(max(Var(3), Var(0)), cos(neg(Var(0)))), mul_add(Var(3), Var(1), Var(0)))), log10(add(abs(neg(Const(-1.1988422))), Const(0.001)))), max(abs(neg(ln(add(abs(neg(pow(add(abs(Var(1)), Const(0.001)), Var(2)))), Const(0.001))))), ln(add(abs(mul(Const(-0.47071946), pow(add(abs(neg(Const(-1.670574))), Const(0.001)), Var(0)))), Const(0.001)))))))), Const(0.001)), Const(-1)))), Const(0.001)))";
        let tree = logged_expr_scalar_output(initial);
        let roundtrip = logged_expr_roundtrip_scalar_output(initial);
        assert!(
            (tree - roundtrip).abs() <= 1e-6,
            "tree: {tree}, roundtrip: {roundtrip}"
        );
    }

    #[test]
    // These deeply-nested expressions exceed the x86-64 JIT's spill-free SSE
    // register budget (it rejects them with an error). The aarch64 backend uses
    // linear-scan allocation with spilling, so it handles arbitrary depth.
    #[cfg(target_arch = "aarch64")]
    fn seed_24042_t52_final_jit_matches_scalar() {
        let final_ = "div(min(add(mul(min(pow(add(abs(neg(neg(abs(neg(neg(add(mul(cos(neg(neg(Var(1)))), Const(0.52093434)), Const(-1.414685)))))))), Const(0.001)), Const(-1)), Var(1)), log10(add(abs(neg(neg(add(add(max(add(mul(Var(2), Var(2)), Var(3)), Var(1)), atan2(atan2(Var(0), Var(3)), add(Var(1), neg(Var(3))))), mul(ln(add(Const(0.12621832), Const(0.001))), mul_add(Var(0), cos(neg(neg(Var(3)))), pow(add(Const(0.20414245), Const(0.001)), Const(-0.5)))))))), Const(0.001)))), pow(add(abs(neg(neg(atan2(Var(2), log10(add(abs(neg(neg(tan(Var(1))))), Const(0.001))))))), Const(0.001)), Const(-0.5))), log2(add(abs(neg(neg(Var(0)))), Const(0.001)))), add(abs(neg(pow(add(abs(neg(pow(add(abs(neg(log2(add(abs(neg(mul(min(add(mul(Const(0.093906365), Var(2)), mul(Const(0.093906365), Var(3))), max(Const(-1.3763785), mul(Var(0), Const(0.5072496)))), mul(pow(abs(neg(neg(log10(add(abs(neg(neg(Var(3)))), Const(0.001)))))), Const(0.5)), ln(add(abs(neg(sin(Var(2)))), Const(0.001))))))), Const(0.001))))), Const(0.001)), mul(add(add(mul(div(add(Const(0.61049294), Const(1.354384)), add(abs(neg(mul(Var(0), Var(0)))), Const(0.001))), mul(Const(8.641348), recip(add(Const(1.7208018), Const(0.001))))), add(mul(max(Var(3), Var(0)), cos(neg(Var(0)))), mul_add(Var(3), Var(1), Var(0)))), log10(add(Const(1.1988422), Const(0.001)))), max(abs(neg(ln(add(abs(neg(pow(add(abs(Var(1)), Const(0.001)), Var(2)))), Const(0.001))))), ln(add(abs(mul(Const(-0.47071946), pow(add(Const(1.670574), Const(0.001)), Var(0)))), Const(0.001)))))))), Const(0.001)), Const(-1)))), Const(0.001)))";
        assert_scalar_and_jit_close(final_, 1e-3);
    }
}
