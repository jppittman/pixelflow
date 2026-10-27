# pixelflow-compiler/src/emit

**Is.** The front end's last stage: the function from an analyzed block to the
tokens that rebuild, at load time, the kernels it denotes — each entry a term of
arrow type with uniform parameters (DENOTATION §2), its arena replayed push by
push — or lowering's refusal.

## Owns
- The stage's function (`host.rs`): `emit_kernel` — per entry, `Expansion::lower`
  or `Staged::lower`, the optimizer's answer when the entry may be read, then the host items: an
  expression for the closure form; a `#[repr(C)]` struct per record, a `const`
  per `pub const`, a `#[must_use]` host `fn` and its `Args` record per entry.
- Which entries an optimizer sees (`entry_arena`): none with a
  kernel-typed parameter; a `Changed` arena relinked to declaration order.
- The arena's replay (`arena.rs`): every node the root reaches, children first,
  each a push; a uniform declared per argument with the call's value as its
  default. Count parameters are uniforms and may bound a fold by value.
- The snippets (`arena.rs`): `emitted!`, the mixed-site `quote!` every emitted
  local is spelled with, so `__arena`, `__t3`, `__lo` never meet a user's name.
- The second `Site` (`staged.rs`): `Staged`, lowering's steps as the statements
  that take them, for entries with kernel parameters.

## Never
- No lowering and no checking: the walk is `lower_entry`'s, run through
  `Expansion` or `Staged`; a node no body can spell (`Buffer`, `Ref`) is a panic.
- No optimization: `optimizer` is a value handed in; `Declined` needs no arm.
- No `::std` path, so a `no_std` crate expands any block; a constant and a
  fold header travel as bits.
- No call's value becomes a constant: every uniform is declared with the call's
  value as its default, so one entry at one instantiation is one program.
- No item but `emit_kernel` leaves the module: every other item is `pub(super)` or private.

## Contract
`mod.rs`, under `# Contract`: `emit_kernel`, implemented in `host.rs` and
`pub(crate) use`d there. One function, `(analyzed, optimizer)` to
`syn::Result<TokenStream>`, deterministic in both.

## Enforced by
- Types: `Lowered` gives emission an arena, a root and its scalars; `FoldHeader`
  crosses as bits with a total inverse, so no emitted code assembles one that
  means nothing. Value bounds are guarded by the IR.
- Visibility: `rg 'pub\(crate\)' pixelflow-compiler/src/emit --glob '!mod.rs'`
  lists `emit_kernel` alone.
- `pixelflow-compiler/src/tests/front_end.rs`: three tests through `front_end` — a
  record is `#[repr(C)]` and no path is `::std`; a template's range is refused at
  instantiation; an optimizer is offered only the entries it can read. `tests/items_block.rs`,
  `tests/binding_times.rs`, `tests/kernel_typed_parameters.rs` compile what is
  emitted. `scripts/check-intent.sh`, `scripts/check-doc-paths.sh`.

## Behind
- `node` (`arena.rs`) and `Staged` (`staged.rs`) are two spellings of "an IR
  call as the statement that makes it"; replaying an arena into a `Site` would
  reduce them to one (the-front-ends-stages §4).
- No `Param(k)` hole reaches the emitted arena.

## Decided in
`docs/compiler/2026-10-22-the-front-ends-stages-are-its-modules.md` (§1.1, §1.2, §1.5, §4),
`docs/language/2026-09-25-the-language-is-kernel.md` (§1.2–§1.4, §1.8, D-a, Phase E), `docs/DENOTATION.md` §2, §8.
