# Denotational Machine Learning & Symbolic Autodiff

## Vision
PixelFlow models Neural Networks as pure mathematical functions (Manifolds). 
To compute gradients for training, we do not use Reverse-Mode AD (tapes) or standard Forward-Mode AD loops. 

Instead, we put **the Calculus inside the E-Graph**. The E-Graph performs fully symbolic differentiation by matching `Derivative(expr)` nodes and rewriting them using the chain rule, resulting in a single, perfectly optimized, flat arithmetic AST. 

## The Architecture

### 1. The Frontend API (Pure Composition)
The user defines layers exactly like graphics formulas.

```rust
// Inputs and Weights are Lattices
let inputs = Lattice::new(...);
let w1 = Lattice::new(...);

// Layer 1
let dot1 = w1.at(Z, X) * inputs.at(Z);
let l1_out = Reduce::new(dot1, Z, 2).max(0.0);

// Loss Function: Mean Squared Error
let diff = target.at(X) - l1_out.at(X);
let loss = Reduce::new(diff * diff, X, 3);
```

### 2. The Derivative Request
The user requests the gradient of the loss. The frontend macro simply wraps the entire AST in a new node: `OpKind::Derivative`.

```rust
// Internally: Expr::Unary(OpKind::Derivative, loss)
let grad_w1 = kernel_jit!(|| loss.derivative_wrt(&w1));
```

### 3. Calculus in the E-Graph
We add calculus rewrite rules directly into `pixelflow-search`. 
When the E-Graph encounters a `Derivative` node, it applies symbolic chain rule substitutions:

- `Derivative(Add(A, B))` -> `Add(Derivative(A), Derivative(B))`
- `Derivative(Mul(A, B))` -> `Add(Mul(A, Derivative(B)), Mul(B, Derivative(A)))`
- `Derivative(Sin(A))` -> `Mul(Cos(A), Derivative(A))`
- `Derivative(Var(Target))` -> `1.0`
- `Derivative(Var(Other))` -> `0.0`

### 4. Learned Traversal vs. Expression Swell
The symbolic derivative of a massive neural network generates an astronomical number of `* 0.0` terms (because differentiating wrt `Weight_A` produces 0 for `Weight_B`).

In a standard Equality Saturation engine (like basic `egg`), applying the product rule factorially explodes the state space ("Expression Swell"), causing the compiler to run out of memory. 

PixelFlow survives this by using **Learned Traversal** (MCTS / NNUE guided search) instead of blind equality saturation. The ML cost models guide the E-Graph to preferentially select the chain rule expansions followed immediately by zero-annihilation reductions (`Mul(X, 0.0) -> 0.0`), aggressively pruning the AST and bypassing the combinatorial explosion entirely.

The E-Graph perfectly factors the equation down to the minimum necessary arithmetic operations, fundamentally generating the optimal math without ever allocating a gradient tape.

### 5. The "Wall" (Fallback to Duals)
If the E-Graph hits an expression it *cannot* symbolically differentiate (an opaque function, a custom black-box shader node, or a `CallExternal`), it hits the "Wall."

At this boundary, the E-Graph simply leaves the `Derivative` node unresolved. 
When the compiler emits the final backend IR, if it sees an unresolved `Derivative(OpaqueNode)`, it emits a fallback instruction to evaluate that specific node using runtime `Dual` numbers.

This provides the ultimate hybrid system:
1. **99% of the network** is analytically differentiated and compiled to pure `f32` math.
2. **1% opaque nodes** fall back gracefully to our existing, mathematically robust Forward-Mode AD system.
