# Kernel staging and rectangular user folds

## 4. Rectangular user-fold bounds

A fold written through `kernel!`, `kernel_raw!`, or `Kernel::over_values` has
bounds that are either constants or unchanged uniform counts. Arithmetic on a
uniform is not a bound, and neither a coordinate nor an enclosing fold index is
a bound. Consequently every user-written nested fold has a rectangular
iteration space.

An entry parameter `n: usize` is the surface spelling of a uniform count. It
may appear unchanged at either end of a range. Constants retain their existing
`usize` arithmetic because that arithmetic is evaluated during expansion.

## Boundary

This is a restriction on user construction, not on the IR's denotation.
`ExprArena` and rewrite rules continue to represent value bounds, including
bounds that read an enclosing binder. Optimizers may therefore discover or
transform non-rectangular folds without making them part of the stable source
or builder API.

Independent nesting remains ordinary: a constant inner range, or one bounded
by an unchanged uniform that does not depend on the outer index, is admitted.

## Diagnostics

The macro rejects an enclosing index at that index's use as a range bound. The
public builder reports `BoundIndex`. Existing refusals continue to distinguish
coordinates, arithmetic, values outside the exact count domain, masks, and
values that are not numeric bounds.
