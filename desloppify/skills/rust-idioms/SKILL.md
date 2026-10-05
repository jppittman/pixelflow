---
name: rust-idioms
description: What idiomatic, unslopped Rust looks like in this codebase.
---

# Rust idioms

- Comments say *why*, never *what*. Unclear code is refactored, not explained.
- Early returns and `let ... else` over nested `if`/`else`.
- `match` over `else if` chains on enums.
- No boolean arguments; use an enum or two functions.
- Errors are handled or propagated with `?`. `let _ =` on a `Result` is a bug.
- A name that stacks concepts (`compile_arena_dag_jet`) is a namespace; it wants
  to be a module, a method, or a builder.
- A second way of doing something the code already does one way is a second
  `impl` of a trait, not a copy or a mode flag.
