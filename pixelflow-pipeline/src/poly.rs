//! The two schedules for a polynomial, as arena builders.
//!
//! A polynomial is a *value*: `p(x) = Σ aᵢ xⁱ`. Horner and Estrin are two
//! *schedules* for that value — same coefficients, same function, different
//! dependency graph. Horner is a chain of `n` fused multiply-adds, each
//! waiting on the last; Estrin is a balanced tree of the same `n` FMAs over a
//! `log₂ n` chain of squarings, so its critical path is `O(log n)` where
//! Horner's is `O(n)`, at the cost of `log₂ n` extra multiplies and a wider
//! live set.
//!
//! Everything in the compiler that prices an expression prices its *ops*, not
//! its *shape* (`CostModel::latency_prior` sums node costs), so the two forms
//! are indistinguishable to extraction except that Estrin looks strictly
//! worse. Whether the machine agrees is an empirical question, which is what
//! `examples/horner_vs_estrin.rs` measures.
//!
//! Both builders emit `OpKind::MulAdd` for every coefficient step, matching
//! what `pixelflow_ir::passes`'s `horner_step` emits for the transcendental
//! expansions — so the Horner arm here IS production's shape, not a
//! restatement of it.

use pixelflow_ir::{ExprBuilder, ExprData, ExprHandle, Node, OpKind};

/// Which schedule to emit for the same coefficient list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolyForm {
    /// `a₀ + x(a₁ + x(a₂ + …))` — one `MulAdd` per coefficient, all serial.
    Horner,
    /// Balanced tree over the powers `x², x⁴, x⁸, …` — the same count of
    /// `MulAdd`s plus one `Mul` per power level, in a `O(log n)`-deep DAG.
    Estrin,
}

impl PolyForm {
    /// Short lowercase name, for table headers and JSONL records.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            PolyForm::Horner => "horner",
            PolyForm::Estrin => "estrin",
        }
    }
}

/// Emit `Σ coeffs[i]·xⁱ` into `arena` under `form`, returning its root.
///
/// `coeffs` is **ascending** in degree (`coeffs[0]` is the constant term),
/// matching `pixelflow_ir::passes::EXP2_POLY` and friends.
///
/// # Panics
///
/// Panics on an empty coefficient list: the degree-0 polynomial is a constant,
/// which has no schedule to compare and is never what a caller meant.
#[must_use]
pub fn build(
    builder: &mut ExprBuilder,
    form: PolyForm,
    coeffs: &[f32],
    x: ExprHandle,
) -> ExprHandle {
    assert!(!coeffs.is_empty(), "poly::build: empty coefficient list");
    match form {
        PolyForm::Horner => horner(builder, coeffs, x),
        PolyForm::Estrin => estrin(builder, coeffs, x),
    }
}

/// `a₀ + x(a₁ + x(a₂ + …))`, highest degree down — one `MulAdd` per step.
fn horner(builder: &mut ExprBuilder, coeffs: &[f32], x: ExprHandle) -> ExprHandle {
    let mut acc = builder.constant(coeffs[coeffs.len() - 1]);
    for &c in coeffs.iter().rev().skip(1) {
        let c = builder.constant(c);
        acc = builder.ternary(OpKind::MulAdd, acc, x, c);
    }
    acc
}

/// Estrin's scheme: pair adjacent coefficients into `aᵢ + x·aᵢ₊₁`, then fold
/// the pairs together against `x²`, `x⁴`, `x⁸`, … until one node is left.
///
/// An odd element at the end of a level is carried up unchanged rather than
/// padded with a zero coefficient — a `MulAdd` against a zero addend is still
/// a real instruction, and the folder cannot remove it (`x·0` is not `0` for
/// non-finite `x`).
fn estrin(builder: &mut ExprBuilder, coeffs: &[f32], x: ExprHandle) -> ExprHandle {
    // Level 0: the linear pairs. `MulAdd(a, b, c)` is `a·b + c`.
    let mut level: Vec<ExprHandle> = coeffs
        .chunks(2)
        .map(|pair| match pair {
            [lo, hi] => {
                let lo = builder.constant(*lo);
                let hi = builder.constant(*hi);
                builder.ternary(OpKind::MulAdd, hi, x, lo)
            }
            [lo] => builder.constant(*lo),
            _ => unreachable!("chunks(2) yields 1 or 2 elements"),
        })
        .collect();

    let mut power = x;
    while level.len() > 1 {
        power = builder.binary(OpKind::Mul, power, power);
        level = level
            .chunks(2)
            .map(|pair| match pair {
                [lo, hi] => builder.ternary(OpKind::MulAdd, *hi, power, *lo),
                [lo] => *lo,
                _ => unreachable!("chunks(2) yields 1 or 2 elements"),
            })
            .collect();
    }
    level[0]
}

/// Interpolate `f` on `[0, 1]` at `n` Chebyshev nodes, returning `n` monomial
/// coefficients ascending — the input [`build`] wants.
///
/// Near-minimax, which is the point: the question "does another term buy
/// accuracy" is only meaningful against a fit that is already as good as a fit
/// of that degree gets. A least-squares fit on equispaced points would answer a
/// question about Runge's phenomenon instead.
///
/// Chebyshev *nodes* with a monomial *basis*: the basis is fixed by what the
/// arena evaluates, and the nodes are what keeps the Vandermonde system solvable
/// there. Conditioning still degrades with degree, which is why callers stop
/// around 14 — well past where `f32` stops rewarding more terms anyway.
///
/// # Panics
///
/// Panics for `n == 0`: a polynomial with no coefficients is not a fit.
#[must_use]
pub fn chebyshev_fit(f: impl Fn(f64) -> f64, n: usize) -> Vec<f32> {
    assert!(n > 0, "chebyshev_fit: need at least one coefficient");
    let nodes: Vec<f64> = (0..n)
        .map(|k| {
            let t = (core::f64::consts::PI * (k as f64 + 0.5) / n as f64).cos();
            (t + 1.0) / 2.0
        })
        .collect();
    let vandermonde: Vec<Vec<f64>> = nodes
        .iter()
        .map(|&x| (0..n).map(|j| x.powi(j as i32)).collect())
        .collect();
    let values: Vec<f64> = nodes.iter().map(|&x| f(x)).collect();
    solve(vandermonde, values)
        .into_iter()
        .map(|c| c as f32)
        .collect()
}

/// Gaussian elimination with partial pivoting. Small, dense, and used once per
/// fit at build time, so the naive algorithm is the right one.
fn solve(mut a: Vec<Vec<f64>>, mut y: Vec<f64>) -> Vec<f64> {
    let n = y.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| {
                a[i][col]
                    .abs()
                    .partial_cmp(&a[j][col].abs())
                    .expect("finite Vandermonde entries")
            })
            .expect("non-empty column");
        a.swap(col, pivot);
        y.swap(col, pivot);
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            let (upper, lower) = a.split_at_mut(row);
            for (target, &source) in lower[0][col..n].iter_mut().zip(&upper[col][col..n]) {
                *target -= factor * source;
            }
            y[row] -= factor * y[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut acc = y[i];
        for k in i + 1..n {
            acc -= a[i][k] * x[k];
        }
        x[i] = acc / a[i][i];
    }
    x
}

/// Longest path from `root` to a leaf, weighted by `cost` — the DAG's
/// critical path, which is what the machine's out-of-order engine is bounded
/// by when consecutive evaluations cannot overlap.
///
/// The sum of the same weights over the same nodes is what
/// `CostModel::latency_prior` charges. Printing both is the point: they are
/// the same table read two ways, and they disagree about Estrin.
#[must_use]
pub fn critical_path(root: Node<'_, ExprData>, cost: impl Fn(OpKind) -> f64) -> f64 {
    let mut memo = std::collections::BTreeMap::<Node<'_, ExprData>, f64>::new();
    // Explicit stack: a polynomial DAG is O(degree) deep, and recursion here
    // would blow the stack for the same reason the degree sweep exists.
    let mut stack = vec![(root, false)];
    while let Some((id, expanded)) = stack.pop() {
        if memo.contains_key(&id) {
            continue;
        }
        if !expanded {
            stack.push((id, true));
            for child in id.children() {
                if !memo.contains_key(&child) {
                    stack.push((child, false));
                }
            }
            continue;
        }
        let kind = match *id {
            ExprData::Var(_) => OpKind::Var,
            ExprData::Const(_) | ExprData::Param(_) => OpKind::Const,
            ExprData::Buffer(_) => OpKind::Buffer,
            ExprData::Uniform(_) => OpKind::Uniform,
            ExprData::Ref(key) => panic!("poly::critical_path: Ref({key:?}) reached analysis"),
            ExprData::Op(op) => op,
            ExprData::Reduce(_) => OpKind::Reduce,
        };
        let own = match kind {
            OpKind::Var | OpKind::Const | OpKind::Buffer => 0.0,
            k => cost(k),
        };
        let deepest_child = id
            .children()
            .map(|c| {
                memo.get(&c)
                    .copied()
                    .expect("children resolved before parent")
            })
            .fold(0.0f64, f64::max);
        memo.insert(id, own + deepest_child);
    }
    memo.get(&root).copied().expect("root resolved")
}

#[cfg(all(test, feature = "training"))]
mod tests {
    use super::*;

    fn coeffs(n: usize) -> Vec<f32> {
        // Alternating, decaying — a well-conditioned polynomial on [0, 1],
        // so a disagreement between the forms is a scheduling bug and not
        // catastrophic cancellation.
        (0..n)
            .map(|i| {
                let sign = if i.is_multiple_of(2) { 1.0 } else { -1.0 };
                sign * 0.75f32.powi(i as i32) / (i as f32 + 1.0)
            })
            .collect()
    }

    /// Estrin buys depth with width: strictly more nodes, strictly shorter
    /// critical path, for every degree past the trivial ones. This is the
    /// structural claim the benchmark then prices — if it ever stops holding,
    /// the builder is wrong and the timings mean nothing.
    #[test]
    fn estrin_is_shallower_and_wider_than_horner() {
        let unit = |_k: OpKind| 1.0;
        for n in 6..=33 {
            let cs = coeffs(n);
            let mut ha = ExprBuilder::new();
            let hx = ha.var(0);
            let h = build(&mut ha, PolyForm::Horner, &cs, hx);
            let hg = ha.finish_one(h);
            let mut ea = ExprBuilder::new();
            let ex = ea.var(0);
            let e = build(&mut ea, PolyForm::Estrin, &cs, ex);
            let eg = ea.finish_one(e);

            let hd = critical_path(hg.root(), unit);
            let ed = critical_path(eg.root(), unit);
            assert!(
                ed < hd,
                "degree {n}: estrin depth {ed} not below horner {hd}"
            );

            assert!(
                op_nodes(hg.root()) < op_nodes(eg.root()),
                "degree {n}: estrin should cost extra multiplies for its powers"
            );
        }
    }

    /// The fit must actually converge on the function, in `f64`, before any
    /// conclusion drawn from "does another term help" means anything. Checked
    /// against `exp2` — the function `EXP2_POLY` approximates — in the
    /// arithmetic the fit is computed in, so this pins the fit and not the
    /// evaluator's rounding.
    #[test]
    fn chebyshev_fit_converges_with_degree() {
        let f = |x: f64| x.exp2();
        let err = |n: usize| {
            let cs = chebyshev_fit(f, n);
            (0..=200)
                .map(|i| {
                    let x = f64::from(i) / 200.0;
                    let mut p = 0.0f64;
                    for &c in cs.iter().rev() {
                        p = p * x + f64::from(c);
                    }
                    (p - f(x)).abs()
                })
                .fold(0.0f64, f64::max)
        };
        // Each added term must buy at least 4x, up to where f32 coefficient
        // storage stops being able to express the improvement.
        let mut prev = err(2);
        for n in 3..=6 {
            let cur = err(n);
            assert!(
                cur < prev / 4.0,
                "degree {n}: error {cur:.3e} did not improve 4x on {prev:.3e}"
            );
            prev = cur;
        }
        assert!(
            prev < 1e-6,
            "degree 6 fit should reach 1e-6, got {prev:.3e}"
        );
    }

    fn op_nodes(root: Node<'_, ExprData>) -> usize {
        let mut visited = std::collections::BTreeSet::new();
        let mut stack = vec![root];
        let mut n = 0;
        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            if !matches!(
                *id,
                ExprData::Var(_) | ExprData::Const(_) | ExprData::Buffer(_)
            ) {
                n += 1;
            }
            stack.extend(id.children());
        }
        n
    }
}
