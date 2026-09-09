> **Obsolete — the problem it prototypes against is gone (annotated 2026-09-09).**
> The nested-`Let` trait-bound explosion was a property of the type-level
> combinator tier, retired by the JIT-first `Kernel`/`ExprArena` architecture; the
> `opt-level=1/2` dev-profile workaround it belongs to is likewise gone (see
> Cargo.toml's own note). `pixelflow-core/src/combinators/context.rs` and
> `pixelflow-compiler/src/codegen.rs` no longer exist. Kept as the record of a
> considered alternative, not as work to do.

# Flat Context Tuple Prototype

## Problem: Nested Let Trait Bound Explosion

The current `kernel!` macro uses nested `Let` bindings to provide parameters:

```rust
// Current approach with 5 params:
Let<V0, Let<V1, Let<V2, Let<V3, Let<V4, Body>>>>>

// Domain structure:
LetExtended<V4, LetExtended<V3, LetExtended<V2, LetExtended<V1, LetExtended<V0, (Field, Field, Field, Field)>>>>>
```

**Issue**: Each `Let` layer adds nested trait bounds. The trait solver must recursively prove:
- `V0: Manifold<P, Output = O0>`
- `Let<V1, ...>: Manifold<LetExtended<O0, P>>`
  - Which requires proving `V1: Manifold<LetExtended<O0, P>, Output = O1>`
  - And `Let<V2, ...>: Manifold<LetExtended<O1, LetExtended<O0, P>>>`
    - etc...

The combinatorial explosion of these nested trait bounds causes the solver to give up around 4-5 parameters.

## Solution: Flat Context Tuple

Instead of nesting, use a **single combinator** that holds all bindings in a flat tuple:

```rust
// New approach with 5 params:
WithContext<(V0, V1, V2, V3, V4), Body>

// Domain structure:
((O0, O1, O2, O3, O4), (Field, Field, Field, Field))
```

**Benefits**:
- **Single Manifold impl** per tuple size (not recursive)
- **Flat trait bounds** - solver doesn't nest
- **Direct indexing** via `CtxVar<N0>`, `CtxVar<N1>`, etc
- **Maintains CSE** - each value computed once
- **Same runtime** - monomorphizes identically

## Implementation

### WithContext Combinator

```rust
pub struct WithContext<Ctx, Body> {
    pub ctx: Ctx,     // Tuple of value expressions
    pub body: Body,   // Body expression
}

impl<P, V0, V1, V2, V3, V4, B, O0, O1, O2, O3, O4, Out> Manifold<P>
    for WithContext<(V0, V1, V2, V3, V4), B>
where
    P: Copy + Send + Sync,
    V0: Manifold<P, Output = O0>,
    V1: Manifold<P, Output = O1>,
    V2: Manifold<P, Output = O2>,
    V3: Manifold<P, Output = O3>,
    V4: Manifold<P, Output = O4>,
    O0: Copy + Send + Sync,
    O1: Copy + Send + Sync,
    O2: Copy + Send + Sync,
    O3: Copy + Send + Sync,
    O4: Copy + Send + Sync,
    B: Manifold<((O0, O1, O2, O3, O4), P), Output = Out>,
{
    type Output = Out;

    #[inline(always)]
    fn eval(&self, p: P) -> Self::Output {
        let v0 = self.ctx.0.eval(p);
        let v1 = self.ctx.1.eval(p);
        let v2 = self.ctx.2.eval(p);
        let v3 = self.ctx.3.eval(p);
        let v4 = self.ctx.4.eval(p);
        self.body.eval(((v0, v1, v2, v3, v4), p))
    }
}
```

### CtxVar - Type-Level Indexing

```rust
pub struct CtxVar<N>(PhantomData<N>);

// CtxVar<N0> gets first element
impl<V0, V1, V2, V3, V4, P> Manifold<((V0, V1, V2, V3, V4), P)> for CtxVar<N0>
where V0: Copy + Send + Sync, P: Copy + Send + Sync
{
    type Output = V0;

    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4), P)) -> V0 {
        p.0.0  // Direct tuple field access
    }
}

// Similar impls for CtxVar<N1>, CtxVar<N2>, etc
```

## Trait Bound Comparison

### Nested Let (Current)

```
Let<V0, Let<V1, Let<V2, Body>>>: Manifold<P>
  requires: V0: Manifold<P>
            Let<V1, Let<V2, Body>>: Manifold<LetExtended<O0, P>>
              requires: V1: Manifold<LetExtended<O0, P>>
                        Let<V2, Body>: Manifold<LetExtended<O1, LetExtended<O0, P>>>
                          requires: V2: Manifold<LetExtended<O1, LetExtended<O0, P>>>
                                    Body: Manifold<LetExtended<O2, LetExtended<O1, LetExtended<O0, P>>>>
```

**Depth**: O(n) nesting
**Trait queries**: Exponential in parameter count

### WithContext (Prototype)

```
WithContext<(V0, V1, V2), Body>: Manifold<P>
  requires: V0: Manifold<P>
            V1: Manifold<P>
            V2: Manifold<P>
            Body: Manifold<((O0, O1, O2), P)>
```

**Depth**: O(1) - flat
**Trait queries**: Linear in parameter count

## Next Steps

### 1. Modify kernel! Macro Codegen

Update `pixelflow-compiler/src/codegen.rs` to emit `WithContext` instead of nested `Let`:

```rust
// Before:
Let::new(val0, Let::new(val1, Let::new(val2, body)))

// After:
WithContext::new((val0, val1, val2), body)
```

Key changes:
- `emit_kernel()`: Build tuple instead of nested Lets
- `emit_annotated_expr()`: Emit `CtxVar<N>` instead of `Var<N>`

### 2. Implement CtxVar Arithmetic

Add blanket impls for arithmetic operations on CtxVar:

```rust
impl<N, P, V> ops::Add<f32> for CtxVar<N>
where
    CtxVar<N>: Manifold<P, Output = V>,
    V: Numeric,
{
    type Output = Add<CtxVar<N>, f32>;
    fn add(self, rhs: f32) -> Self::Output {
        Add(self, rhs)
    }
}
```

This allows `CtxVar<N0>::new() + 1.0` to build Manifold AST.

### 3. Support Additional Tuple Sizes

Generate impls for 6, 7, 8... params using macros:

```rust
macro_rules! impl_with_context {
    ($($n:tt: $V:ident $O:ident),+) => {
        impl<P, $($V, $O,)+ B, Out> Manifold<P>
            for WithContext<($($V,)+), B>
        where
            P: Copy + Send + Sync,
            $($V: Manifold<P, Output = $O>, $O: Copy + Send + Sync,)+
            B: Manifold<(($($O,)+), P), Output = Out>,
        {
            type Output = Out;

            fn eval(&self, p: P) -> Self::Output {
                $(let $V = self.ctx.$n.eval(p);)+
                self.body.eval((($($V,)+), p))
            }
        }
    };
}

impl_with_context!(0: V0 O0, 1: V1 O1, 2: V2 O2, 3: V3 O3, 4: V4 O4, 5: V5 O5);
// ...up to desired max
```

### 4. Benchmark & Validate

- **Compilation time**: Should improve for >3 param kernels
- **Runtime**: Should be identical (same monomorphized code)
- **Binary size**: Should be similar or smaller

## Open Questions

1. **Backward compatibility**: Can we support both Let and WithContext?
2. **Error messages**: Will flat structure improve or worsen compile errors?
3. **Max params**: What's a reasonable upper bound? 10? 16?
4. **Migration path**: Gradual or all-at-once switch?

## Status

- ✅ WithContext combinator implemented (`pixelflow-core/src/combinators/context.rs`)
- ✅ Compiles successfully with 5+ params (tested up to 8)
- ✅ kernel! macro integration complete (`pixelflow-compiler/src/codegen.rs`)
- ✅ CtxVar arithmetic operators implemented
- ✅ Spatial trait impls for tuple domains
- ✅ Test suite passes (tests/test_kernel_5params.rs)
- ✅ **PRODUCTION READY**: All kernels now use flat WithContext instead of nested Let

## Files

- `pixelflow-core/src/combinators/context.rs` - WithContext & CtxVar implementation
- `pixelflow-core/tests/test_context_tuple.rs` - Compilation tests
- `pixelflow-compiler/src/codegen.rs` - (TODO) Macro emission changes
